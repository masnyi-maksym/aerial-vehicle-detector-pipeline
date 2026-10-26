"""Stage 03: fuse the three labelers' boxes into one set with votes k (1..3).

Per frame:
    1. keep each model's boxes with conf >= its own ``vote_conf`` (score scales differ),
    2. greedy clustering, highest conf first: a box joins the best-matching cluster
       that has no box from its model yet, otherwise starts a new cluster.
       Near pairs (< 200 m) match by IoU, far pairs by NWD (IoU collapses on tiny boxes),
    3. fused box = conf-weighted mean of the cluster's boxes (WBF logic), k = number of models,
    4. flag "container" boxes that hold >= 2 other boxes (group boxes, truck + trailer),
    5. drop k=1 boxes nested inside a k>=2 non-container box (truck cab / trailer parts,
       duplicates); nested boxes with k>=2 are kept and flagged for review.

Outputs (outputs/03_fuse/):
    fused.parquet   clip_id, frame_idx, x1..y2, k, models, conf_<model>..., conf_mean,
                    dist_m, band, container, nested
    stats.json      k distribution per clip and band, which model produces lone (k=1) boxes
    viz/            random frames, box colour = k (green 3, yellow 2, red 1), magenta = container

Examples:
    python -m pipeline.s03_fuse
    python -m pipeline.s03_fuse --iou-only       # ablation: what NWD adds for far pairs (scripts/fuse_ablation.py)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from pipeline.common import Stage, base_arg_parser, load_clips, load_yaml, start_stage
from pipeline.geometry import contained_fraction, distance_m, iou_one_to_many, nwd_one_to_many

LOGGER = logging.getLogger("pipeline.s03_fuse")

K_COLORS_BGR = {3: (60, 170, 60), 2: (0, 200, 255), 1: (40, 40, 230)}  # green, yellow, red
CONTAINER_BGR = (200, 60, 200)


def load_predictions(predict_dir: Path, models: dict[str, Any]) -> pd.DataFrame:
    """Loads every model's raw predictions from stage 02.

    Args:
        predict_dir: Folder with ``{model}.parquet``.
        models: ``predict.models`` config section.

    Returns:
        Concatenated predictions with a ``model`` column.

    Raises:
        FileNotFoundError: If a model's predictions are missing.
    """
    parts = []
    for name in models:
        path = predict_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run stage 02 for model '{name}' first")
        parts.append(pd.read_parquet(path).assign(model=name))
    return pd.concat(parts, ignore_index=True)


def select_votes(preds: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """Keeps each model's boxes at or above that model's vote threshold.

    Args:
        preds: Raw predictions with a ``model`` column.
        thresholds: ``{model: vote_conf}``.

    Returns:
        Votes.
    """
    return preds[preds["conf"] >= preds["model"].map(thresholds)]


def processed_frames(preds: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Catalog rows of every frame stage 02 ran on, including frames without detections.

    Stage 02 writes no row for a frame where a model found nothing, so frames are
    taken from the union of all models' predictions.

    Args:
        preds: Raw predictions of all models.
        catalog: Full frame catalog.

    Returns:
        Catalog rows sorted by clip and frame.
    """
    keys = preds[["clip_id", "frame_idx"]].drop_duplicates()
    return catalog.merge(keys, on=["clip_id", "frame_idx"]).sort_values(["clip_id", "frame_idx"], ignore_index=True)


def match_scores(box: np.ndarray, fused: np.ndarray, f_px: float, cfg: dict[str, Any],
                 ref_length_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Similarity of one box to each cluster box: IoU for near pairs, NWD for far pairs.

    Args:
        box: ``(4,)`` candidate box.
        fused: ``(M, 4)`` current cluster boxes.
        f_px: Focal length of the frame in pixels.
        cfg: ``fuse`` config section.
        ref_length_m: Reference vehicle length in metres.

    Returns:
        ``(scores, passed)``: similarity per cluster and whether it clears its threshold.
    """
    pair_dist = (distance_m(box[None], f_px, ref_length_m) + distance_m(fused, f_px, ref_length_m)) / 2
    near = pair_dist < float(cfg["near_max_m"])
    iou = iou_one_to_many(box, fused)
    nwd = nwd_one_to_many(box, fused, float(cfg["nwd_c"]))
    scores = np.where(near, iou, nwd)
    passed = np.where(near, iou >= float(cfg["iou_thr"]), nwd >= float(cfg["nwd_thr"]))
    return scores, passed


def fuse_frame(votes: pd.DataFrame, model_names: list[str], f_px: float, cfg: dict[str, Any],
               ref_length_m: float) -> pd.DataFrame:
    """Greedy clustering of one frame's votes into fused boxes.

    Args:
        votes: Boxes of one frame (columns ``x1..y2, conf, model``).
        model_names: All model keys (for per-model confidence columns).
        f_px: Focal length of the frame in pixels.
        cfg: ``fuse`` config section.
        ref_length_m: Reference vehicle length in metres.

    Returns:
        Fused boxes with ``k``, ``models`` and ``conf_<model>`` columns.
    """
    order = np.argsort(-votes["conf"].to_numpy(), kind="stable")
    boxes = votes[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)[order]
    confs = votes["conf"].to_numpy(dtype=np.float64)[order]
    models = votes["model"].to_numpy()[order]

    fused: list[np.ndarray] = []
    members: list[dict[str, tuple[np.ndarray, float]]] = []
    for box, conf, model in zip(boxes, confs, models):
        if fused:
            scores, passed = match_scores(box, np.asarray(fused), f_px, cfg, ref_length_m)
            passed &= np.array([model not in m for m in members])
            if passed.any():
                j = int(np.argmax(np.where(passed, scores, -np.inf)))
                members[j][model] = (box, conf)
                w = np.array([c for _, c in members[j].values()])
                fused[j] = (np.stack([b for b, _ in members[j].values()]) * w[:, None]).sum(0) / w.sum()
                continue
        members.append({model: (box, conf)})
        fused.append(box.copy())

    rows = []
    for box, mem in zip(fused, members):
        row: dict[str, Any] = dict(zip(["x1", "y1", "x2", "y2"], box.tolist()))
        row["k"] = len(mem)
        row["models"] = "+".join(sorted(mem))
        for name in model_names:
            row[f"conf_{name}"] = mem[name][1] if name in mem else np.nan
        row["conf_mean"] = float(np.mean([c for _, c in mem.values()]))
        rows.append(row)
    return pd.DataFrame(rows)


def containment_flags(boxes: np.ndarray, k: np.ndarray, frac: float, min_inside: int) -> tuple[np.ndarray, np.ndarray]:
    """Finds container boxes and boxes nested inside a confirmed box.

    * container: holds >= ``min_inside`` other boxes (Grounding DINO group boxes,
      whole-truck boxes over a separately detected tractor and trailer);
    * nested: lies inside a larger box that has k >= 2 and is not a container
      (truck cab, trailer, a shifted duplicate).

    Args:
        boxes: ``(N, 4)`` fused boxes of one frame.
        k: ``(N,)`` votes per box.
        frac: Share of a box that must lie inside another to count as inside.
        min_inside: How many contained boxes make a container.

    Returns:
        ``(container, nested)`` boolean masks.
    """
    n = len(boxes)
    inside = np.zeros((n, n), dtype=bool)  # inside[i, j]: box j lies inside box i
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for i in range(n):
        inside[i] = (contained_fraction(boxes[i], boxes) >= frac) & (areas < areas[i])
    container = inside.sum(axis=1) >= min_inside
    parents = (k >= 2) & ~container
    nested = (inside & parents[:, None]).any(axis=0)
    return container, nested


def band_labels(dist: np.ndarray, edges: list[float]) -> list[str]:
    """Maps distances to band names like ``0-200``, ``200-400``, ``>400``.

    Args:
        dist: Distances in metres.
        edges: Increasing band edges, e.g. ``[0, 200, 400]``.

    Returns:
        Band name per distance.
    """
    names = [f"{int(a)}-{int(b)}" for a, b in zip(edges[:-1], edges[1:])] + [f">{int(edges[-1])}"]
    idx = np.clip(np.searchsorted(edges, dist, side="right") - 1, 0, len(names) - 1)
    return [names[i] for i in idx]


def fuse_all(votes: pd.DataFrame, frames: pd.DataFrame, model_names: list[str], cfg: dict[str, Any],
             ref_length_m: float) -> tuple[pd.DataFrame, int]:
    """Fuses every frame and applies the containment rules.

    Args:
        votes: Votes of all models (after per-model thresholds).
        frames: Catalog rows of the processed frames (need ``f_px``).
        model_names: All model keys.
        cfg: ``fuse`` config section.
        ref_length_m: Reference vehicle length in metres.

    Returns:
        Fused boxes and the number of dropped nested lone boxes.
    """
    parts = []
    by_frame = dict(list(votes.groupby(["clip_id", "frame_idx"], sort=False)))
    edges = [float(e) for e in cfg["bands_m"]]
    for row in frames.itertuples(index=False):
        frame_votes = by_frame.get((row.clip_id, row.frame_idx))
        if frame_votes is None:
            continue
        out = fuse_frame(frame_votes, model_names, float(row.f_px), cfg, ref_length_m)
        boxes = out[["x1", "y1", "x2", "y2"]].to_numpy()
        out["dist_m"] = distance_m(boxes, float(row.f_px), ref_length_m)
        out["band"] = band_labels(out["dist_m"].to_numpy(), edges)
        out["container"], out["nested"] = containment_flags(
            boxes, out["k"].to_numpy(), float(cfg["container_frac"]), int(cfg["container_min"]))
        out.insert(0, "frame_idx", row.frame_idx)
        out.insert(0, "clip_id", row.clip_id)
        parts.append(out)

    fused = pd.concat(parts, ignore_index=True)
    drop = fused["nested"] & (fused["k"] == 1) if cfg["drop_nested_lone"] else pd.Series(False, index=fused.index)
    return fused[~drop].reset_index(drop=True), int(drop.sum())


def per_band_k(fused: pd.DataFrame, n_frames: int) -> dict[str, dict[str, float]]:
    """Boxes per frame by band and k.

    Args:
        fused: Fused boxes (any subset of clips).
        n_frames: Number of frames the boxes come from.

    Returns:
        ``{band: {"k3": .., "k2": .., "k1": ..}}``.
    """
    return {band: {f"k{k}": round(float((g["k"] == k).sum()) / n_frames, 2) for k in (3, 2, 1)}
            for band, g in fused.groupby("band")}


def compute_stats(fused: pd.DataFrame, frames: pd.DataFrame, n_dropped: int) -> dict[str, Any]:
    """Summary of votes per clip and band, and which model produces lone boxes.

    Args:
        fused: All fused boxes.
        frames: Catalog rows of the processed frames.
        n_dropped: Nested lone boxes removed by ``fuse_all``.

    Returns:
        JSON-serialisable stats.
    """
    stats: dict[str, Any] = {}
    n_frames = frames.groupby("clip_id").size()
    for clip_id, group in fused.groupby("clip_id"):
        n = int(n_frames[clip_id])
        stats[clip_id] = {
            "frames": n,
            "boxes_per_frame": round(len(group) / n, 1),
            "k_share": {f"k{k}": round(float((group["k"] == k).mean()), 3) for k in (3, 2, 1)},
            "k_per_frame_by_band": per_band_k(group, n),
            "lone_boxes_by_model": group.loc[group["k"] == 1, "models"].value_counts().to_dict(),
            "containers_per_frame": round(float(group["container"].sum()) / n, 2),
            "nested_kept_per_frame": round(float(group["nested"].sum()) / n, 2),
        }
    stats["nested_lone_dropped"] = n_dropped
    stats["frames_without_boxes"] = int(len(frames) - fused[["clip_id", "frame_idx"]].drop_duplicates().shape[0])
    return stats


def render_viz(stage: Stage, fused: pd.DataFrame, frames: pd.DataFrame, datasets_root: Path, n: int) -> None:
    """Draws fused boxes coloured by k on random frames.

    Args:
        stage: Stage handle.
        fused: All fused boxes.
        frames: Catalog rows of the processed frames.
        datasets_root: Root for catalog paths.
        n: Number of frames to render.
    """
    rng = np.random.default_rng(int(stage.cfg["seed"]))
    picks = frames.iloc[np.sort(rng.choice(len(frames), size=min(n, len(frames)), replace=False))]
    for row in picks.itertuples(index=False):
        img = cv2.imread(str(datasets_root / row.path))
        scale = min(1.0, 1920 / max(img.shape[:2]))
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        sel = fused[(fused["clip_id"] == row.clip_id) & (fused["frame_idx"] == row.frame_idx)]
        for r in sel.itertuples(index=False):
            p1 = (int(r.x1 * scale), int(r.y1 * scale))
            p2 = (int(r.x2 * scale), int(r.y2 * scale))
            if r.container:
                cv2.rectangle(img, p1, p2, CONTAINER_BGR, 3)
            cv2.rectangle(img, p1, p2, K_COLORS_BGR[int(r.k)], 2)
        counts = sel["k"].value_counts()
        legend = [(f"k=3: {counts.get(3, 0)}", K_COLORS_BGR[3]), (f"k=2: {counts.get(2, 0)}", K_COLORS_BGR[2]),
                  (f"k=1: {counts.get(1, 0)}", K_COLORS_BGR[1]),
                  (f"container: {int(sel['container'].sum())}", CONTAINER_BGR)]
        cv2.rectangle(img, (0, 0), (300, 12 + 30 * len(legend)), (255, 255, 255), -1)
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
    parser.add_argument("--predict-dir", type=Path, default=None, help="Stage 02 output folder (default: from config).")
    parser.add_argument("--iou-only", action="store_true",
                        help="Ablation: match every pair by IoU (no NWD for far pairs); writes outputs/03_fuse_iou_only.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    stage = start_stage("03_fuse_iou_only" if args.iou_only else "03_fuse", cfg, args)
    fcfg, models = dict(cfg["fuse"]), cfg["predict"]["models"]
    if args.iou_only:
        fcfg["near_max_m"] = float("inf")  # every pair counts as near -> IoU
    predict_dir = args.predict_dir or Path(fcfg["predict_dir"])
    camera, _ = load_clips(Path(cfg["paths"]["clips"]))

    preds = load_predictions(predict_dir, models)
    frames = processed_frames(preds, pd.read_parquet(cfg["paths"]["catalog"]))
    votes = select_votes(preds, {name: float(m["vote_conf"]) for name, m in models.items()})
    LOGGER.info("Fusing %d frames from %s, votes per model: %s", len(frames), predict_dir,
                votes["model"].value_counts().to_dict())

    fused, n_dropped = fuse_all(votes, frames, list(models), fcfg, float(camera["ref_length_m"]))
    fused.to_parquet(stage.out_dir / "fused.parquet", index=False)
    LOGGER.info("Fused boxes: %d (dropped %d nested lone), k: %s", len(fused), n_dropped,
                fused["k"].value_counts().sort_index().to_dict())
    stage.save_stats(compute_stats(fused, frames, n_dropped))
    render_viz(stage, fused, frames, Path(cfg["paths"]["datasets"]), int(fcfg["viz_frames"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
