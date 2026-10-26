"""Stage 04b: second look at every track on context crops - vehicle, two-wheeler or nothing.

Why: stage 02 asked the detectors only for car / bus / truck, so a motorcycle had no better
class to go to and sometimes came out as a "car"; the zoom check (03b) only re-checked small
doubtful boxes. Here every track gets a few context crops (evenly spaced observed boxes;
crop side = ``context`` x the box long side, but at least input / ``max_scale`` px so a tiny
box is upscaled at most 2.5x, and big boxes are downscaled) and YOLO + RT-DETR look at them with
two-wheeler classes added (COCO bicycle, motorcycle). A detection counts for the track box if it
matches it as in fusion (IoU near, NWD far). Classes compete: per crop and model the best
matching vehicle and two-wheeler confidences decide the verdict of that check.

Rotation: every crop is also checked turned by 90 deg (``rotations``). Two questions in one run:
r0 on a crop that holds the whole object vs. the original tile pass separates "cut by a tile
edge" from "not seen"; r90 vs r0 shows whether orientation matters (COCO cars are mostly seen
from the side, horizontal). The TTA verdict takes the best confidence over rotations.

Per track (recheck.parquet):
    rc_n             number of checks (crops x models)
    rc_vehicle       share of checks: a vehicle class matched and beat two-wheelers
    rc_two_wheeler   share of checks: bicycle / motorcycle matched and beat vehicles
    rc_none          share of checks: nothing matched
    rc_vehicle_conf  mean best matching vehicle confidence (0 if none)
    rc_vehicle_r<d>  vehicle share using rotation d only
    long_px, orient  box long side and orientation (horizontal / vertical / square)

Outputs (outputs/04b_recheck/):
    checks.parquet   one row per crop and model
    recheck.parquet  one row per track
    stats.json       verdict shares per track flag and clip
    viz/two_wheelers.jpg   tracks most often called bicycle / motorcycle
    viz/rejected.jpg       non-flicker tracks no check called a vehicle, per flag
    viz/rescued.jpg        flicker tracks most checks called a vehicle
    viz/rotation.jpg       tracks only the turned crop called a vehicle (per orientation), and the reverse

Example:
    python -m pipeline.s04b_recheck
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from pipeline.common import base_arg_parser, load_clips, load_yaml, start_stage
from pipeline.detectors import Detections, build_detector
from pipeline.s02_predict import iter_images
from pipeline.s03_fuse import match_scores
from pipeline.s03b_zoom import crop_window
from pipeline.viz import crop_sheet

LOGGER = logging.getLogger("pipeline.s04b_recheck")
KEYS = ["clip_id", "track_id"]


def sample_checks(boxes: pd.DataFrame, n: int) -> pd.DataFrame:
    """Evenly spaced observed boxes of every track (interpolated boxes are skipped).

    Args:
        boxes: Tracked boxes (stage 04).
        n: Boxes per track (fewer if the track is shorter).

    Returns:
        Selected boxes.
    """
    obs = boxes[~boxes["interpolated"].astype(bool)].sort_values([*KEYS, "frame_idx"])
    pos = obs.groupby(KEYS).cumcount()
    size = obs.groupby(KEYS)["frame_idx"].transform("size")
    keep = np.zeros(len(obs), dtype=bool)
    for m in range(n):  # positions round(m * (size - 1) / (n - 1)), deduplicated by the mask
        target = (m * (size - 1) / max(n - 1, 1)).round().astype(int)
        keep |= (pos == target).to_numpy()
    return obs[keep]


def best_by_group(d: Detections, window: tuple[int, int, int], input_px: int, box: np.ndarray, f_px: float,
                  fuse_cfg: dict[str, Any], ref_length_m: float, groups: dict[str, list[str]]) -> dict[str, float]:
    """Best confidence per class group among detections matching the track box.

    Args:
        d: Detections of one crop (crop pixels).
        window: ``(x0, y0, side)`` of the crop in frame pixels.
        input_px: Size the crop was resized to.
        box: Track box in frame pixels.
        f_px: Focal length of the frame.
        fuse_cfg: ``fuse`` config section (matching rule).
        ref_length_m: Reference vehicle length.
        groups: ``{group: [class names]}``.

    Returns:
        ``{group: confidence}`` (0 if no matching detection of that group).
    """
    out = {g: 0.0 for g in groups}
    if len(d.scores) == 0:
        return out
    x0, y0, side = window
    mapped = d.boxes.astype(np.float64) * (side / input_px) + np.array([x0, y0, x0, y0])
    _, passed = match_scores(box, mapped, f_px, fuse_cfg, ref_length_m)
    labels = np.array(d.labels)
    for g, names in groups.items():
        sel = passed & np.isin(labels, names)
        if sel.any():
            out[g] = float(d.scores[sel].max())
    return out


def unrotate(d: Detections, k: int, size: int) -> Detections:
    """Maps boxes found on ``np.rot90(crop, k)`` back to the unrotated square crop.

    Args:
        d: Detections on the rotated crop.
        k: Number of counter-clockwise quarter turns that were applied (0-3).
        size: Side of the square crop, px.

    Returns:
        Detections in unrotated crop pixels.
    """
    b = d.boxes.astype(np.float64)
    if k % 4 == 0 or len(b) == 0:
        return d
    x1, y1, x2, y2 = b.T
    if k % 4 == 1:    # counter-clockwise quarter turn: x = size - y', y = x'
        out = np.stack([size - y2, x1, size - y1, x2], axis=1)
    elif k % 4 == 2:  # x = size - x', y = size - y'
        out = np.stack([size - x2, size - y2, size - x1, size - y1], axis=1)
    else:             # x = y', y = size - x'
        out = np.stack([y1, size - x2, y2, size - x1], axis=1)
    return Detections(boxes=out.astype(np.float32), scores=d.scores, labels=d.labels)


def run_checks(checks: pd.DataFrame, catalog: pd.DataFrame, datasets_root: Path, cfg: dict[str, Any],
               full_cfg: dict[str, Any], ref_length_m: float) -> pd.DataFrame:
    """Runs every re-check model on the context crops of the selected boxes.

    Frames are read once; crops of consecutive frames are batched together.

    Args:
        checks: Selected boxes.
        catalog: Frame catalog indexed by ``(clip_id, frame_idx)``.
        datasets_root: Root for catalog paths.
        cfg: ``recheck`` config section.
        full_cfg: Full pipeline config (model sections, fuse section, device).
        ref_length_m: Reference vehicle length.

    Returns:
        One row per check: clip, track, frame, model, rotation, best confidence per group.
    """
    rotations = [int(r) for r in cfg.get("rotations", [0])]
    if any(r % 90 for r in rotations):
        raise ValueError(f"rotations must be multiples of 90, got {rotations}")
    pcfg = full_cfg["predict"]
    dets = {}
    for name in cfg["models"]:
        mcfg = dict(pcfg["models"][name], classes=list(cfg["classes"]), conf=float(cfg["conf"]))
        if mcfg["type"] not in {"yolo", "rtdetr"}:
            raise ValueError(f"recheck supports Ultralytics models only, got {name}")
        dets[name] = build_detector(name, mcfg, pcfg["device"])
    groups = {g: list(v) for g, v in cfg["groups"].items()}
    batch = max(d.batch for d in dets.values())
    frames = list(checks.groupby(["clip_id", "frame_idx"], sort=True))
    paths = [datasets_root / catalog.loc[key, "path"] for key, _ in frames]

    records: list[dict[str, Any]] = []
    pending: list[tuple[Any, np.ndarray, float, dict[int, tuple[int, int, int]]]] = []
    crops: dict[int, list[np.ndarray]] = {}

    def flush() -> None:
        for name, det in dets.items():
            out: list[Detections] = []
            for s in range(0, len(crops[det.tile]), det.batch):
                out += det.predict(crops[det.tile][s:s + det.batch])
            it = iter(out)
            for row, box, f_px, windows in pending:
                for rot in rotations:  # crops were appended rotation by rotation for every box
                    d = unrotate(next(it), rot // 90, det.tile)
                    conf = best_by_group(d, windows[det.tile], det.tile, box, f_px, full_cfg["fuse"], ref_length_m,
                                         groups)
                    records.append({"clip_id": row.clip_id, "track_id": row.track_id, "frame_idx": row.frame_idx,
                                    "model": name, "rot": rot, **{f"{g}_conf": c for g, c in conf.items()}})
        pending.clear()
        crops.clear()

    t0 = time.perf_counter()
    for i, (((clip_id, frame_idx), g), img) in enumerate(zip(frames, iter_images(paths))):
        h, w = img.shape[:2]
        f_px = float(catalog.loc[(clip_id, frame_idx), "f_px"])
        for row in g.itertuples(index=False):
            box = np.array([row.x1, row.y1, row.x2, row.y2], dtype=np.float64)
            long_side = max(box[2] - box[0], box[3] - box[1], 1.0)
            windows = {}
            for tile in {d.tile for d in dets.values()}:
                scale = min(float(cfg["max_scale"]), tile / (float(cfg["context"]) * long_side))
                x0, y0, side = windows[tile] = crop_window(box, scale, tile, w, h)
                interp = cv2.INTER_CUBIC if side < tile else cv2.INTER_AREA
                crop = cv2.resize(img[y0:y0 + side, x0:x0 + side], (tile, tile), interpolation=interp)
                for rot in rotations:
                    crops.setdefault(tile, []).append(np.ascontiguousarray(np.rot90(crop, rot // 90)))
            pending.append((row, box, f_px, windows))
        if len(pending) >= batch:
            flush()
        if (i + 1) % 500 == 0:
            LOGGER.info("%d / %d frames, %.0f s", i + 1, len(frames), time.perf_counter() - t0)
    if pending:
        flush()
    LOGGER.info("%d checks on %d frames, %.0f s", len(records), len(frames), time.perf_counter() - t0)
    return pd.DataFrame(records)


def track_verdicts(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregates checks into per-track shares (see module docstring).

    Args:
        results: One row per check and rotation with ``vehicle_conf`` and ``two_wheeler_conf``.

    Returns:
        One row per track: TTA shares (``rc_vehicle`` ...) and ``rc_vehicle_r<deg>`` per rotation.
    """
    def shares(r: pd.DataFrame) -> pd.DataFrame:
        veh, tw = r["vehicle_conf"], r["two_wheeler_conf"]
        verdict = np.select([(veh > 0) & (veh >= tw), tw > veh], ["vehicle", "two_wheeler"], "none")
        out = pd.crosstab([r["clip_id"], r["track_id"]], verdict, normalize="index")
        return out.reindex(columns=["vehicle", "two_wheeler", "none"], fill_value=0.0)

    # all rotations together (test-time augmentation): best confidence per group over rotations
    tta = results.groupby([*KEYS, "frame_idx", "model"])[["vehicle_conf", "two_wheeler_conf"]].max().reset_index()
    out = tta.groupby(KEYS).agg(rc_n=("model", "size"), rc_vehicle_conf=("vehicle_conf", "mean"))
    out = out.join(shares(tta).add_prefix("rc_"))
    for rot, r in results.groupby("rot"):
        out[f"rc_vehicle_r{rot}"] = shares(r)["vehicle"]
    return out.reset_index()


def track_geometry(checks: pd.DataFrame) -> pd.DataFrame:
    """Orientation and size of every track from its checked boxes (frame pixels).

    Args:
        checks: Selected boxes.

    Returns:
        ``clip_id, track_id, long_px, orient`` (horizontal: w >= 1.25 h, vertical: h >= 1.25 w,
        else square, i.e. diagonal or compact) and ``size_bin`` of the long side.
    """
    g = checks.assign(w=checks["x2"] - checks["x1"], h=checks["y2"] - checks["y1"]).groupby(KEYS)[["w", "h"]].median()
    g["long_px"] = g[["w", "h"]].max(axis=1)
    g["orient"] = np.select([g["w"] >= 1.25 * g["h"], g["h"] >= 1.25 * g["w"]], ["horizontal", "vertical"], "square")
    g["size_bin"] = pd.cut(g["long_px"], [0, 32, 64, 128, np.inf], labels=["<32", "32-64", "64-128", ">128"])
    return g[["long_px", "orient", "size_bin"]].reset_index()


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    parser = base_arg_parser(__doc__)
    parser.add_argument("--input-path", type=Path, default=None, help="Stage 04 tracked.parquet (default: from config).")
    parser.add_argument("--max-tracks", type=int, default=0, help="Re-check only this many random tracks (speed test).")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    stage = start_stage("04b_recheck", cfg, args)
    rcfg = cfg["recheck"]
    camera, _ = load_clips(Path(cfg["paths"]["clips"]))
    datasets_root = Path(cfg["paths"]["datasets"])
    catalog = pd.read_parquet(cfg["paths"]["catalog"]).set_index(["clip_id", "frame_idx"])
    boxes = pd.read_parquet(args.input_path or Path(rcfg["input_path"]))
    if args.max_tracks > 0:
        keys = boxes[KEYS].drop_duplicates()
        keys = keys.sample(min(args.max_tracks, len(keys)), random_state=int(cfg["seed"]))
        boxes = boxes.merge(keys, on=KEYS)

    checks = sample_checks(boxes, int(rcfg["crops_per_track"]))
    LOGGER.info("%d tracks, %d crops x %d models", checks[KEYS].drop_duplicates().shape[0], len(checks),
                len(rcfg["models"]))
    results = run_checks(checks, catalog, datasets_root, rcfg, cfg, float(camera["ref_length_m"]))
    results.to_parquet(stage.out_dir / "checks.parquet", index=False)

    tracks = track_verdicts(results)
    flags = boxes.groupby(KEYS)["track_flag"].first().rename("flag").reset_index()
    tracks = tracks.merge(flags, on=KEYS, how="left").merge(track_geometry(checks), on=KEYS, how="left")
    tracks.to_parquet(stage.out_dir / "recheck.parquet", index=False)

    rc_cols = ["rc_vehicle", "rc_two_wheeler", "rc_none", "rc_vehicle_conf"]
    by_flag = tracks.groupby("flag")[rc_cols].mean().round(3)
    stats = {
        "tracks": len(tracks),
        "mean_shares_by_flag": by_flag.to_dict(orient="index"),
        "two_wheeler_majority_by_clip": tracks[tracks["rc_two_wheeler"] >= 0.5]["clip_id"].value_counts().to_dict(),
        "no_vehicle_at_all_by_flag": tracks[tracks["rc_vehicle"] == 0]["flag"].value_counts().to_dict(),
        "per_model_vehicle_share": results.assign(v=results["vehicle_conf"] > results["two_wheeler_conf"])
                                          .groupby(["model", "rot"])["v"].mean().round(3).unstack().to_dict(),
    }
    rot_cols = [c for c in tracks.columns if c.startswith("rc_vehicle_r")]
    if len(rot_cols) > 1:  # does turning the crop change what the detectors see?
        for name, sel in (("stable_moving", tracks["flag"] == "stable_moving"), ("all", tracks["flag"].notna())):
            table = tracks[sel].groupby(["orient", "size_bin"], observed=True)[rot_cols].mean().round(3)
            table["tracks"] = tracks[sel].groupby(["orient", "size_bin"], observed=True).size()
            stats[f"rotation_{name}"] = {f"{o} {b}": row.to_dict() for (o, b), row in table.iterrows()}
            with pd.option_context("display.width", 200):
                LOGGER.info("Vehicle share per rotation, %s tracks (orientation x long side px):\n%s",
                            name, table.to_string())
    stage.save_stats(stats)
    with pd.option_context("display.width", 200):
        LOGGER.info("Mean verdict shares per track flag:\n%s", by_flag.to_string())
    LOGGER.info("Two-wheeler majority tracks per clip: %s", stats["two_wheeler_majority_by_clip"])
    LOGGER.info("Tracks no check called a vehicle, per flag: %s", stats["no_vehicle_at_all_by_flag"])

    obs = checks.merge(catalog[["path"]].reset_index(), on=["clip_id", "frame_idx"])
    rng = np.random.default_rng(int(cfg["seed"]))
    cols = int(rcfg["viz_per_row"])

    def caption(t: Any) -> str:
        return f"v{t.rc_vehicle:.2f} 2w{t.rc_two_wheeler:.2f} {t.flag[:3]}"

    def sample(t: pd.DataFrame) -> pd.DataFrame:
        return t.iloc[rng.permutation(len(t))[:cols]]

    tw = tracks[tracks["rc_two_wheeler"] > 0].sort_values("rc_two_wheeler", ascending=False)
    crop_sheet([(f"two-wheeler\ntop {i * cols + 1}-{(i + 1) * cols}", tw.iloc[i * cols:(i + 1) * cols])
                for i in range(4)], obs, datasets_root, stage.viz_dir / "two_wheelers.jpg", cols, caption)
    rejected = tracks[(tracks["rc_vehicle"] == 0) & (tracks["flag"] != "flicker")]
    crop_sheet([(f"no vehicle\n{f} ({(rejected['flag'] == f).sum()})", sample(rejected[rejected["flag"] == f]))
                for f in ("stable_moving", "stable_static", "short")], obs, datasets_root,
               stage.viz_dir / "rejected.jpg", cols, caption)
    rescued = tracks[(tracks["rc_vehicle"] >= 0.5) & (tracks["flag"] == "flicker")]
    n_rescued, rescued = len(rescued), rescued.iloc[rng.permutation(len(rescued))[:2 * cols]]
    crop_sheet([(f"flicker, vehicle\n>= 0.5 ({n_rescued})", rescued.iloc[:cols]), ("", rescued.iloc[cols:])],
               obs, datasets_root, stage.viz_dir / "rescued.jpg", cols, caption)
    if len(rot_cols) > 1:
        r0, r1 = rot_cols[0], rot_cols[1]
        gain = tracks[(tracks[r0] == 0) & (tracks[r1] >= 0.5)]
        loss = tracks[(tracks[r0] >= 0.5) & (tracks[r1] == 0)]
        rows = [(f"{r1[-3:]} only\n{o} ({(gain['orient'] == o).sum()})", sample(gain[gain["orient"] == o]))
                for o in ("horizontal", "vertical", "square")]
        rows.append((f"{r0[-2:]} only\nall ({len(loss)})", sample(loss)))
        crop_sheet(rows, obs, datasets_root, stage.viz_dir / "rotation.jpg", cols,
                   lambda t: f"{t.long_px:.0f}px {getattr(t, r0):.2f}>{getattr(t, r1):.2f} {t.flag[:3]}")
    LOGGER.info("Viz -> %s", stage.viz_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
