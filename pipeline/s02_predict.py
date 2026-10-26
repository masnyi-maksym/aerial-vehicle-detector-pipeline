"""Stage 02: raw predictions of every labeler, saved unfiltered for offline fusion.

For each model and frame:
    1. cut the frame into tiles of the model's native input size (overlap 20%),
    2. drop tile boxes cut by an inner tile edge (the object is whole in a neighbour tile),
    3. add one pass on the whole frame for large near vehicles,
    4. class-agnostic NMS over tile + full-frame boxes of that model.
A low confidence threshold is used on purpose: vote thresholds are tuned later
(stage 03) on cached predictions without re-running the models.

Outputs (outputs/02_predict/):
    {model}.parquet   clip_id, frame_idx, x1, y1, x2, y2, conf, src_class, pass
    stats.json        boxes per model/clip, seconds per frame, full-run time estimates
    viz/              random frames with all models' boxes (conf >= vote_conf)

Examples:
    python -m pipeline.s02_predict                              # full run
    python -m pipeline.s02_predict --models gdino_base          # re-run one model
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from pipeline.common import Stage, base_arg_parser, frame_steps, load_clips, load_yaml, start_stage
from pipeline.detectors import Detector, build_detector
from pipeline.geometry import nms, tile_windows, touches_inner_edge

LOGGER = logging.getLogger("pipeline.s02_predict")

COLUMNS = ["clip_id", "frame_idx", "x1", "y1", "x2", "y2", "conf", "src_class", "pass"]
VIZ_COLORS_BGR = [(183, 74, 83), (86, 110, 15), (48, 90, 216)]  # purple, teal, coral


def select_frames(catalog: pd.DataFrame, steps: dict[str, int], clips: list[str] | None = None) -> pd.DataFrame:
    """Picks frames to label: every ``step``-th frame of each clip (see ``common.frame_steps``).

    Args:
        catalog: Frame catalog from stage 01.
        steps: ``{clip_id: step}``; keeps ``frame_idx % step == 0`` (clips not listed: every frame).
        clips: Optional clip IDs to restrict to.

    Returns:
        Selected catalog rows sorted by clip and frame.
    """
    sel = catalog[catalog["frame_idx"] % catalog["clip_id"].map(steps).fillna(1).astype(int) == 0]
    if clips:
        sel = sel[sel["clip_id"].isin(clips)]
    return sel.sort_values(["clip_id", "frame_idx"], ignore_index=True)


def iter_images(paths: list[Path], prefetch: int = 4) -> Iterator[np.ndarray]:
    """Reads images in a background thread, keeping at most ``prefetch`` in memory.

    Args:
        paths: Image files in order.
        prefetch: Number of images decoded ahead.

    Yields:
        BGR images in the same order.

    Raises:
        FileNotFoundError: If an image cannot be read.
    """
    def read(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return img

    with ThreadPoolExecutor(max_workers=2) as pool:
        queue: deque[Future[np.ndarray]] = deque(pool.submit(read, p) for p in paths[:prefetch])
        for nxt in paths[prefetch:] + [None] * prefetch:
            if not queue:
                break
            img = queue.popleft().result()
            if nxt is not None:
                queue.append(pool.submit(read, nxt))
            yield img


def predict_frame(det: Detector, img: np.ndarray, cfg: dict[str, Any]) -> pd.DataFrame:
    """Tiled + full-frame inference of one model on one frame.

    Args:
        det: Detector.
        img: BGR frame at native resolution.
        cfg: ``predict`` section of the pipeline config.

    Returns:
        Boxes in frame pixels with columns ``x1..y2, conf, src_class, pass``.
    """
    height, width = img.shape[:2]
    windows = tile_windows(width, height, det.tile, float(cfg["tile_overlap"]))
    boxes, scores, labels, passes = [], [], [], []

    for start in range(0, len(windows), det.batch):
        chunk = windows[start:start + det.batch]
        crops = [np.ascontiguousarray(img[y0:y1, x0:x1]) for x0, y0, x1, y1 in chunk]
        for window, dets in zip(chunk, det.predict(crops)):
            keep = ~touches_inner_edge(dets.boxes, window, (width, height), float(cfg["edge_margin_px"]))
            offset = np.array([window[0], window[1], window[0], window[1]], dtype=np.float32)
            boxes.append(dets.boxes[keep] + offset)
            scores.append(dets.scores[keep])
            labels += [lbl for lbl, k in zip(dets.labels, keep) if k]
            passes += ["tile"] * int(keep.sum())

    if cfg["full_frame_pass"]:
        dets = det.predict([img])[0]
        boxes.append(dets.boxes)
        scores.append(dets.scores)
        labels += dets.labels
        passes += ["full"] * len(dets.labels)

    all_boxes = np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32)
    all_scores = np.concatenate(scores) if scores else np.zeros(0, np.float32)
    keep = nms(all_boxes, all_scores, float(cfg["nms_iou"])) if len(all_scores) else np.zeros(0, int)
    out = pd.DataFrame(all_boxes[keep], columns=["x1", "y1", "x2", "y2"])
    out["conf"] = all_scores[keep]
    out["src_class"] = [labels[i] for i in keep]
    out["pass"] = [passes[i] for i in keep]
    return out


def merge_by_clip(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    """Replaces the clips present in ``new`` inside an existing parquet (partial re-runs).

    Args:
        path: Existing parquet file (may not exist).
        new: Fresh rows.

    Returns:
        Combined rows.
    """
    if not path.exists():
        return new
    old = pd.read_parquet(path)
    return pd.concat([old[~old["clip_id"].isin(new["clip_id"].unique())], new], ignore_index=True)


def run_model(name: str, mcfg: dict[str, Any], frames: pd.DataFrame, cfg: dict[str, Any],
              datasets_root: Path, out_dir: Path, partial: bool) -> dict[str, Any]:
    """Runs one model over the selected frames and writes ``{name}.parquet``.

    Args:
        name: Model key.
        mcfg: Model config section.
        frames: Selected catalog rows.
        cfg: ``predict`` section of the pipeline config.
        datasets_root: Root for catalog paths.
        out_dir: Stage output folder.
        partial: Merge into an existing parquet instead of replacing it.

    Returns:
        Per-clip stats of this model.
    """
    import torch

    det = build_detector(name, mcfg, cfg["device"])
    LOGGER.info("%s: %d frames, tile %d px", name, len(frames), det.tile)
    parts, seconds = [], []
    paths = [datasets_root / p for p in frames["path"]]
    t_prev = time.perf_counter()
    for i, (row, img) in enumerate(zip(frames.itertuples(index=False), iter_images(paths))):
        boxes = predict_frame(det, img, cfg)
        boxes.insert(0, "frame_idx", row.frame_idx)
        boxes.insert(0, "clip_id", row.clip_id)
        parts.append(boxes)
        now = time.perf_counter()
        seconds.append(now - t_prev)
        t_prev = now
        if (i + 1) % 100 == 0:
            LOGGER.info("%s: %d/%d frames, %.2f s/frame", name, i + 1, len(frames), float(np.median(seconds[-100:])))

    preds = pd.concat(parts, ignore_index=True)[COLUMNS] if parts else pd.DataFrame(columns=COLUMNS)
    path = out_dir / f"{name}.parquet"
    (merge_by_clip(path, preds) if partial else preds).to_parquet(path, index=False)
    LOGGER.info("%s: %d boxes -> %s", name, len(preds), path)

    timing = frames.assign(sec=seconds)
    stats = {}
    for clip_id, group in timing.groupby("clip_id"):
        clip_preds = preds[preds["clip_id"] == clip_id]
        n_vote = int((clip_preds["conf"] >= float(mcfg["vote_conf"])).sum())
        stats[clip_id] = {
            "frames": len(group),
            "boxes": len(clip_preds),
            "boxes_per_frame": round(len(clip_preds) / len(group), 1),
            "boxes_per_frame_above_vote_conf": round(n_vote / len(group), 1),
            "sec_per_frame_median": round(float(group["sec"].iloc[1:].median() if len(group) > 1 else group["sec"].iloc[0]), 3),
        }

    del det
    torch.cuda.empty_cache()
    return stats


def estimate_minutes(model_stats: dict[str, Any], catalog: pd.DataFrame, steps: dict[str, int]) -> float:
    """Estimates a full run (all clips) from measured seconds per frame.

    Args:
        model_stats: Per-clip stats with ``sec_per_frame_median``.
        catalog: Full frame catalog.
        steps: Frame step per clip to estimate for.

    Returns:
        Minutes, counting only clips that were measured.
    """
    full = select_frames(catalog, steps).groupby("clip_id").size()
    sec = sum(model_stats[c]["sec_per_frame_median"] * int(full[c]) for c in model_stats if c in full)
    return round(sec / 60, 1)


def render_viz(stage: Stage, frames: pd.DataFrame, cfg: dict[str, Any], datasets_root: Path) -> None:
    """Draws all models' boxes (conf >= vote_conf) on random frames for a quick visual check.

    Args:
        stage: Stage handle (output folder, seed).
        frames: Selected catalog rows.
        cfg: ``predict`` section of the pipeline config.
        datasets_root: Root for catalog paths.
    """
    models = cfg["models"]
    files = {name: stage.out_dir / f"{name}.parquet" for name in models}
    if not all(p.exists() for p in files.values()):
        LOGGER.info("Viz skipped: not all models have predictions yet")
        return
    preds = {name: pd.read_parquet(p) for name, p in files.items()}
    rng = np.random.default_rng(int(stage.cfg["seed"]))
    picks = frames.iloc[np.sort(rng.choice(len(frames), size=min(int(cfg["viz_frames"]), len(frames)), replace=False))]

    for row in picks.itertuples(index=False):
        img = cv2.imread(str(datasets_root / row.path))
        scale = min(1.0, 1920 / max(img.shape[:2]))
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        legend = []
        for i, (name, df) in enumerate(preds.items()):
            color = VIZ_COLORS_BGR[i % len(VIZ_COLORS_BGR)]
            sel = df[(df["clip_id"] == row.clip_id) & (df["frame_idx"] == row.frame_idx)
                     & (df["conf"] >= float(models[name]["vote_conf"]))]
            for x1, y1, x2, y2 in (sel[["x1", "y1", "x2", "y2"]].to_numpy() * scale).astype(int):
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1 + i)  # thickness differs per model
            legend.append((f"{name}: {len(sel)}", color))
        cv2.rectangle(img, (0, 0), (330, 12 + 30 * len(legend)), (255, 255, 255), -1)
        for i, (text, color) in enumerate(legend):
            cv2.putText(img, text, (10, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imwrite(str(stage.viz_dir / f"{row.clip_id}_{row.frame_idx:06d}.jpg"), img)
    LOGGER.info("Viz: %d frames -> %s", len(picks), stage.viz_dir)


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    parser = base_arg_parser(__doc__)
    parser.add_argument("--models", nargs="*", default=None, help="Only these model keys (default: all).")
    parser.add_argument("--clips", nargs="*", default=None, help="Only these clip IDs (default: all).")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    stage = start_stage("02_predict", cfg, args)
    pcfg = cfg["predict"]
    datasets_root = Path(cfg["paths"]["datasets"])
    catalog = pd.read_parquet(cfg["paths"]["catalog"])
    _, clips = load_clips(Path(cfg["paths"]["clips"]))
    steps = frame_steps(clips, int(pcfg["train_frame_step"]))
    frames = select_frames(catalog, steps, args.clips)
    LOGGER.info("Selected %d frames (steps %s): %s", len(frames), steps,
                frames.groupby("clip_id").size().to_dict())

    stats_path = stage.out_dir / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    partial = bool(args.clips)
    for name in args.models or list(pcfg["models"]):
        model_stats = run_model(name, pcfg["models"][name], frames, pcfg, datasets_root, stage.out_dir, partial)
        stats[name] = {
            "clips": model_stats,
            "est_full_run_min": {"configured_steps": estimate_minutes(model_stats, catalog, steps),
                                 "every_frame": estimate_minutes(model_stats, catalog, {c: 1 for c in steps})},
        }
        LOGGER.info("%s: estimated full run %s min", name, stats[name]["est_full_run_min"])
        stage.save_stats(stats)  # after every model, so an interrupted run keeps finished models' stats
    render_viz(stage, frames, pcfg, datasets_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
