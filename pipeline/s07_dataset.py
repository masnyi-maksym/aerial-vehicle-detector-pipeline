"""Stage 07: YOLO dataset from the auto labels - 640 px tiles at native resolution.

Why tiles: a car at 200-400 m is 12-25 px on 1080p and 25-50 px on 4K; resizing the whole frame
to 640 would shrink it below what a detector can see. Tiles are cut exactly like the auto-labeller
and the evaluation slice the frame (``tile``, ``overlap``), so train and test see the same scale.

Labels (train / val clips): boxes of tracks stage 05 kept as vehicles (tiers A, B, C), including
the boxes the tracker interpolated over short gaps; double boxes (two on one car, a cab inside a
truck box, a group box) are removed with the stage 06 rule. Suspect tracks and two-wheelers stay
unlabelled, i.e. background.

Tiles: every tile with a vehicle; a seeded share of empty tiles (roofs, fields, markings are
useful negatives). A box cut by the tile border is kept (clipped) if at least ``min_visible`` of
it is inside, else dropped. Which empty tiles stay is a hash of the tile name and the seed, so the
same tile gets the same decision in every run and for every frame step.

Rotation cycle (``rotate_cycle``): in each train list the k-th frame of a clip (k = frame_idx //
step) is turned by (k mod 4) x 90 deg: 0, 90, 180, 270, 0, ... No copies, the tile count stays the
same, every orientation gets a quarter of the frames, and neighbouring (near-duplicate) frames look
different. Stage 04b showed COCO detectors miss small vertical cars (0.30 vs 0.74 for horizontal).
Val is never turned.

Frame steps: frames are cut once; ``train_step<s>.txt`` lists the frames with frame_idx % s == 0 in
the rotation of that list (a frame used by two steps with different rotations is written twice).

Outputs:
    datasets/yolo/images/{train,val}/<clip>_<frame>_<x0>_<y0>[_r90|_r180|_r270].jpg, labels/{train,val}/*.txt
    datasets/yolo/train_step<s>.txt, data_step<s>.yaml (val = the val clip at val_step)
    outputs/07_dataset/stats.json, viz/tiles.jpg (random train tiles with their labels)

Example:
    python -m pipeline.s07_dataset
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml

from pipeline.common import base_arg_parser, frame_steps, load_clips, load_yaml, start_stage
from pipeline.geometry import tile_windows
from pipeline.s02_predict import iter_images, select_frames
from pipeline.s06_import import merge_duplicates

LOGGER = logging.getLogger("pipeline.s07_dataset")


def label_boxes(boxes: pd.DataFrame, tiers: pd.DataFrame, cfg: dict[str, Any], import_cfg: dict[str, Any]) -> pd.DataFrame:
    """Vehicle boxes of the auto labels, one per vehicle and frame.

    Args:
        boxes: Tracked boxes (stage 04).
        tiers: Stage 05 track table (``clip_id, track_id, tier``).
        cfg: ``dataset`` config section.
        import_cfg: ``import`` config section (double-box thresholds).

    Returns:
        ``clip_id, frame_idx, track_id, vehicle_id, x1..y2`` of the label boxes.
    """
    keep = tiers.loc[tiers["tier"].isin(cfg["vehicle_tiers"]), ["clip_id", "track_id"]]
    lab = boxes.merge(keep, on=["clip_id", "track_id"])[["clip_id", "frame_idx", "track_id", "x1", "y1", "x2", "y2"]]
    parts = []
    for clip_id, g in lab.groupby("clip_id", sort=True):
        kept, _ = merge_duplicates(g.drop(columns="clip_id").assign(source="file"), float(import_cfg["duplicate_iou"]),
                                   float(import_cfg["part_inside"]))
        parts.append(kept.drop(columns="source").assign(clip_id=clip_id))
    return pd.concat(parts, ignore_index=True) if parts else lab.assign(vehicle_id=pd.Series(dtype=int))


def tile_labels(boxes: np.ndarray, window: tuple[int, int, int, int], min_visible: float) -> np.ndarray:
    """Boxes of one tile in tile pixels, clipped; boxes mostly outside the tile are dropped.

    Args:
        boxes: ``(N, 4)`` frame boxes.
        window: ``(x0, y0, x1, y1)`` tile window.
        min_visible: Minimum share of a box inside the tile.

    Returns:
        ``(M, 4)`` boxes in tile coordinates.
    """
    x0, y0, x1, y1 = window
    clipped = np.stack([np.clip(boxes[:, 0], x0, x1), np.clip(boxes[:, 1], y0, y1),
                        np.clip(boxes[:, 2], x0, x1), np.clip(boxes[:, 3], y0, y1)], axis=1)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    vis = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1]) / np.maximum(area, 1e-6)
    out = clipped[vis >= min_visible] - np.array([x0, y0, x0, y0])
    return out[(out[:, 2] - out[:, 0] >= 2) & (out[:, 3] - out[:, 1] >= 2)]


def stable_uniform(key: str, seed: int) -> float:
    """Deterministic number in [0, 1) for a key: same tile -> same choice, independent of order."""
    return int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 2 ** 32


def rotate_boxes(boxes: np.ndarray, k: int, size: int) -> np.ndarray:
    """Boxes of a square tile after ``np.rot90(tile, k)`` (k counter-clockwise quarter turns).

    Args:
        boxes: ``(N, 4)`` xyxy boxes in tile pixels.
        k: Quarter turns, 0-3.
        size: Tile side.

    Returns:
        Turned boxes.
    """
    if not len(boxes) or k % 4 == 0:
        return boxes
    x1, y1, x2, y2 = boxes.T
    if k % 4 == 1:  # x' = y, y' = size - x
        return np.stack([y1, size - x2, y2, size - x1], axis=1)
    if k % 4 == 2:  # x' = size - x, y' = size - y
        return np.stack([size - x2, size - y2, size - x1, size - y1], axis=1)
    return np.stack([size - y2, x1, size - y1, x2], axis=1)  # k = 3: x' = size - y, y' = x


def frame_rotations(frame_idx: int, steps: list[int], cycle: bool) -> set[int]:
    """Quarter turns a train frame needs: one per frame step whose list contains it.

    Args:
        frame_idx: Frame index in the clip.
        steps: Frame steps of the train lists.
        cycle: Rotation cycle on (else always 0).

    Returns:
        Set of k (0-3).
    """
    return {(frame_idx // s) % 4 if cycle else 0 for s in steps if frame_idx % s == 0}


def write_tile(img: np.ndarray, boxes: np.ndarray, image_path: Path, label_path: Path, quality: int) -> None:
    """Writes one tile image and its YOLO label file (class 0 = vehicle).

    Args:
        img: Tile image.
        boxes: ``(N, 4)`` boxes in tile pixels.
        image_path: JPEG path.
        label_path: Label path (empty file for a negative tile).
        quality: JPEG quality.
    """
    h, w = img.shape[:2]
    cv2.imwrite(str(image_path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    lines = [f"0 {(b[0] + b[2]) / 2 / w:.6f} {(b[1] + b[3]) / 2 / h:.6f} {(b[2] - b[0]) / w:.6f} {(b[3] - b[1]) / h:.6f}"
             for b in boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def cut_split(split: str, frames: pd.DataFrame, labels: pd.DataFrame, root: Path, datasets_root: Path,
              cfg: dict[str, Any], seed: int, rotations: dict[tuple[str, int], set[int]]) -> pd.DataFrame:
    """Cuts and writes the tiles of one split.

    Args:
        split: ``train`` or ``val``.
        frames: Frames to tile (``clip_id, frame_idx, path``), sorted.
        labels: Label boxes of these clips.
        root: Dataset root (``datasets/yolo``).
        datasets_root: Root of the frame paths.
        cfg: ``dataset`` config section.
        seed: Seed of the per-tile choice of empty tiles.
        rotations: ``(clip_id, frame_idx) -> quarter turns to write`` (missing: as is).

    Returns:
        One row per written tile: ``image, clip_id, frame_idx, boxes, rot`` (quarter turns).
    """
    img_dir, lab_dir = root / "images" / split, root / "labels" / split
    for d in (img_dir, lab_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    by_frame = {k: g[["x1", "y1", "x2", "y2"]].to_numpy() for k, g in labels.groupby(["clip_id", "frame_idx"])}
    tile, rows = int(cfg["tile"]), []
    for fr, img in zip(frames.itertuples(index=False), iter_images([datasets_root / p for p in frames["path"]])):
        boxes = by_frame.get((fr.clip_id, fr.frame_idx), np.zeros((0, 4)))
        for window in tile_windows(img.shape[1], img.shape[0], tile, float(cfg["overlap"])):
            tb = tile_labels(boxes, window, float(cfg["min_visible"])) if len(boxes) else np.zeros((0, 4))
            stem = f"{fr.clip_id}_{fr.frame_idx:06d}_{window[0]}_{window[1]}"
            if not len(tb) and stable_uniform(f"{stem}:empty", seed) >= float(cfg["empty_tile_share"]):
                continue
            crop = np.ascontiguousarray(img[window[1]:window[3], window[0]:window[2]])
            for k in sorted(rotations.get((fr.clip_id, fr.frame_idx), {0})):
                if k and crop.shape[0] != crop.shape[1]:
                    continue  # only square tiles can be turned (frames are always >= one tile)
                im = np.ascontiguousarray(np.rot90(crop, k)) if k else crop
                suffix = f"_r{90 * k}" if k else ""
                write_tile(im, rotate_boxes(tb, k, crop.shape[0]), img_dir / f"{stem}{suffix}.jpg",
                           lab_dir / f"{stem}{suffix}.txt", int(cfg["jpeg_quality"]))
                rows.append({"image": str((img_dir / f"{stem}{suffix}.jpg").resolve()), "clip_id": fr.clip_id,
                             "frame_idx": fr.frame_idx, "boxes": len(tb), "rot": k})
    return pd.DataFrame(rows)


def render_tiles(tiles: pd.DataFrame, out: Path, rng: np.random.Generator, n: int = 24, thumb: int = 320) -> None:
    """Contact sheet of random train tiles with their YOLO labels drawn (visual check of the export).

    Args:
        tiles: Written train tiles.
        out: Output JPEG.
        rng: Random generator.
        n: Number of tiles.
        thumb: Tile size on the sheet.
    """
    pick = tiles[tiles["boxes"] > 0]
    pick = pick.iloc[rng.permutation(len(pick))[:n]]
    cols = 6
    sheet = np.full((((len(pick) + cols - 1) // cols) * thumb, cols * thumb, 3), 255, np.uint8)
    for i, t in enumerate(pick.itertuples(index=False)):
        img = cv2.imread(t.image)
        h, w = img.shape[:2]
        lab = Path(t.image.replace("/images/", "/labels/")).with_suffix(".txt").read_text().split("\n")
        for line in filter(None, lab):
            _, cx, cy, bw, bh = (float(v) for v in line.split())
            cv2.rectangle(img, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                          (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (60, 200, 60), 2)
        small = cv2.resize(img, (thumb, thumb), interpolation=cv2.INTER_AREA)
        cv2.putText(small, Path(t.image).stem[-28:], (4, thumb - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        r, c = divmod(i, cols)
        sheet[r * thumb:(r + 1) * thumb, c * thumb:(c + 1) * thumb] = small
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    parser = base_arg_parser(__doc__)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    stage = start_stage("07_dataset", cfg, args)
    dcfg = cfg["dataset"]
    _, clips = load_clips(Path(cfg["paths"]["clips"]))
    roles = {c.clip_id: c.role for c in clips}
    datasets_root = Path(cfg["paths"]["datasets"])
    root = Path(dcfg["out_dir"])
    rng = np.random.default_rng(int(cfg["seed"]))

    boxes = pd.read_parquet(dcfg["boxes_path"])
    tiers = pd.read_parquet(dcfg["tiers_path"])
    labels = label_boxes(boxes, tiers, dcfg, cfg["import"])
    catalog = pd.read_parquet(cfg["paths"]["catalog"])
    frames = select_frames(catalog, frame_steps(clips, int(cfg["predict"]["train_frame_step"])),
                           [c.clip_id for c in clips if c.role in ("train", "val")])
    steps = sorted(int(s) for s in dcfg["train_steps"])
    base = steps[0]
    train_frames = frames[frames["clip_id"].map(roles).eq("train") & (frames["frame_idx"] % base == 0)]
    val_frames = frames[frames["clip_id"].map(roles).eq("val") & (frames["frame_idx"] % int(dcfg["val_step"]) == 0)]

    seed = int(cfg["seed"])
    rotations = {(c, f): frame_rotations(int(f), steps, bool(dcfg["rotate_cycle"]))
                 for c, f in zip(train_frames["clip_id"], train_frames["frame_idx"])}
    train = cut_split("train", train_frames, labels, root, datasets_root, dcfg, seed, rotations)
    val = cut_split("val", val_frames, labels, root, datasets_root, dcfg, seed, {})
    stats: dict[str, Any] = {"label_boxes": int(len(labels)), "val": {
        "frames": len(val_frames), "tiles": len(val), "empty_tiles": int((val["boxes"] == 0).sum()),
        "boxes": int(val["boxes"].sum())}}
    for s in steps:
        sub = train[(train["frame_idx"] % s == 0)
                    & (train["rot"] == ((train["frame_idx"] // s) % 4 if dcfg["rotate_cycle"] else 0))]
        (root / f"train_step{s}.txt").write_text("\n".join(sub["image"]) + "\n", encoding="utf-8")
        data = {"path": str(root.resolve()), "train": f"train_step{s}.txt", "val": "images/val",
                "names": {0: "vehicle"}}
        (root / f"data_step{s}.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        stats[f"train_step{s}"] = {"frames": int(sub[["clip_id", "frame_idx"]].drop_duplicates().shape[0]),
                                   "tiles": len(sub), "tiles_by_rotation_deg": {int(90 * k): int(n) for k, n in
                                                                                 sub["rot"].value_counts().sort_index().items()},
                                   "empty_tiles": int((sub["boxes"] == 0).sum()), "boxes": int(sub["boxes"].sum()),
                                   "by_clip": sub.groupby("clip_id").size().to_dict()}
        LOGGER.info("step %d: %s", s, {k: v for k, v in stats[f"train_step{s}"].items() if k != "by_clip"})
    LOGGER.info("val: %s", stats["val"])
    stage.save_stats(stats)
    if len(train):
        render_tiles(train, stage.viz_dir / "tiles.jpg", rng)
    LOGGER.info("Dataset -> %s (data_step<s>.yaml)", root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
