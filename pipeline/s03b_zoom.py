"""Stage 03b: zoom-in re-check of doubtful small boxes (k <= 2).

A 12-25 px car covers ~1.5-3 cells of the stride-8 feature map, far below the
object sizes COCO models were trained on. Upscaling adds no information but moves
the car into a familiar scale. For every doubtful small box and every zoom model:
    1. cut a square crop around it so that, resized to the model's input size, the box
       long side becomes ~``target_px`` (scale ``min_scale``..``max_scale``),
    2. run the model on the crop; take the highest-scoring detection at the same place
       (same IoU/NWD matching as stage 03) with conf >= the model's vote_conf,
    3. if its label is a vehicle -> zoom vote; if it is a Grounding DINO decoy
       ("traffic cone", "motorcycle", ...) -> recorded as evidence against a vehicle.
       On a crop GDINO sees the decoys next to "car" in one prompt, so the classes
       compete for the box instead of every look-alike becoming a "car".
    4. only a vote from a model that did NOT vote at native scale is new evidence:
       ``k_zoom = k + zoom_new_votes``.
No new boxes are created. A missing zoom vote does not mean "not a car".

Outputs (outputs/03b_zoom/):
    fused_zoom.parquet  fused boxes + zoom_checked, zoom_conf_<model>, zoom_new_votes, k_zoom, zoom_decoy
    stats.json          gained-vote rate and decoy labels by k, band and model
    viz/crops.jpg       contact sheet of random checked crops (green = gained a vote, red = not)

Example:
    python -m pipeline.s03b_zoom
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

from pipeline.common import Stage, base_arg_parser, load_clips, load_yaml, start_stage
from pipeline.detectors import Detector, build_detector, is_target
from pipeline.s02_predict import iter_images
from pipeline.s03_fuse import match_scores

LOGGER = logging.getLogger("pipeline.s03b_zoom")


def zoom_candidates(fused: pd.DataFrame, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Selects doubtful boxes small enough to benefit from upscaling.

    Args:
        fused: Fused boxes from stage 03.
        cfg: ``zoom`` config section.

    Returns:
        ``(mask, scale)``: which boxes to check and the upscale factor of each.
    """
    long_side = np.maximum(fused["x2"] - fused["x1"], fused["y2"] - fused["y1"]).to_numpy()
    scale = np.clip(float(cfg["target_px"]) / np.maximum(long_side, 1e-6), 1.0, float(cfg["max_scale"]))
    mask = (fused["k"].to_numpy() <= int(cfg["max_k"])) & (scale >= float(cfg["min_scale"]))
    return mask, scale


def crop_window(box: np.ndarray, scale: float, input_px: int, width: int, height: int) -> tuple[int, int, int]:
    """Square crop centred on the box, shifted (not cut) to stay inside the image.

    Args:
        box: ``(4,)`` xyxy box in frame pixels.
        scale: Upscale factor for this box.
        input_px: Size the crop is resized to (the model's input size).
        width: Frame width.
        height: Frame height.

    Returns:
        ``(x0, y0, side)`` of the crop in frame pixels.
    """
    side = min(int(round(input_px / scale)), width, height)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0 = int(min(max(round(cx - side / 2), 0), width - side))
    y0 = int(min(max(round(cy - side / 2), 0), height - side))
    return x0, y0, side


def zoom_frame(det: Detector, img: np.ndarray, boxes: np.ndarray, scales: np.ndarray, vote_conf: float,
               f_px: float, fuse_cfg: dict[str, Any], ref_length_m: float) -> tuple[np.ndarray, list[str]]:
    """Re-checks the candidate boxes of one frame with one model.

    Args:
        det: Detector; crops are resized to its native input ``det.tile``.
        img: BGR frame.
        boxes: ``(N, 4)`` candidate boxes.
        scales: ``(N,)`` upscale factors.
        vote_conf: The model's vote threshold.
        f_px: Focal length of the frame in pixels.
        fuse_cfg: ``fuse`` config section (matching thresholds).
        ref_length_m: Reference vehicle length in metres.

    Returns:
        Per candidate: confidence of the best matching detection (NaN if none) and its label.
    """
    input_px = det.tile
    height, width = img.shape[:2]
    windows = [crop_window(b, s, input_px, width, height) for b, s in zip(boxes, scales)]
    crops = [cv2.resize(img[y0:y0 + side, x0:x0 + side], (input_px, input_px), interpolation=cv2.INTER_CUBIC)
             for x0, y0, side in windows]
    dets = []
    for start in range(0, len(crops), det.batch):
        dets += det.predict(crops[start:start + det.batch])

    best = np.full(len(boxes), np.nan)
    labels = [""] * len(boxes)
    for i, ((x0, y0, side), d, box) in enumerate(zip(windows, dets, boxes)):
        keep = np.flatnonzero(d.scores >= vote_conf)
        if keep.size == 0:
            continue
        mapped = d.boxes[keep].astype(np.float64) * (side / input_px) + np.array([x0, y0, x0, y0])
        _, passed = match_scores(box, mapped, f_px, fuse_cfg, ref_length_m)
        if passed.any():
            j = keep[passed][np.argmax(d.scores[keep][passed])]  # classes compete: highest score wins
            best[i], labels[i] = float(d.scores[j]), d.labels[j]
    return best, labels


def compute_stats(checked: pd.DataFrame, models: list[str]) -> dict[str, Any]:
    """Gained-vote rates and decoy labels of the checked boxes by k, band and clip.

    Args:
        checked: Checked boxes.
        models: Zoom model keys.

    Returns:
        JSON-serialisable stats.
    """
    def rates(g: pd.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n": len(g),
            "gained_vote": round(float((g["zoom_new_votes"] > 0).mean()), 3),   # new evidence
            "confirmed_any": round(float((g["zoom_votes"] > 0).mean()), 3),    # incl. self re-confirmation
            "decoy": round(float((g["zoom_decoy"] != "").mean()), 3),
        }
        for m in models:
            out[f"confirmed_{m}"] = round(float(g[f"zoom_conf_{m}"].notna().mean()), 3)
        out["decoy_labels"] = g.loc[g["zoom_decoy"] != "", "zoom_decoy"].value_counts().to_dict()
        return out

    return {
        "all": rates(checked),
        "by_k": {f"k{k}": rates(g) for k, g in checked.groupby("k")},
        "by_band": {band: rates(g) for band, g in checked.groupby("band")},
        "by_clip": {clip: {f"k{k}": rates(gk) for k, gk in g.groupby("k")} for clip, g in checked.groupby("clip_id")},
    }


def render_viz(stage: Stage, checked: pd.DataFrame, catalog: pd.DataFrame, datasets_root: Path,
               cfg: dict[str, Any]) -> None:
    """Contact sheet of random checked crops: green = gained a vote, red = not.

    Args:
        stage: Stage handle.
        checked: Checked boxes.
        catalog: Frame catalog indexed by ``(clip_id, frame_idx)``.
        datasets_root: Root for catalog paths.
        cfg: ``zoom`` config section.
    """
    n = min(int(cfg["viz_crops"]), len(checked))
    if n == 0:
        return
    rng = np.random.default_rng(int(stage.cfg["seed"]))
    picks = checked.iloc[np.sort(rng.choice(len(checked), size=n, replace=False))]
    thumb, cols, context_px = 240, 6, 640
    sheet = np.full((thumb * ((n + cols - 1) // cols), thumb * cols, 3), 255, np.uint8)
    for i, r in enumerate(picks.itertuples(index=False)):
        img = cv2.imread(str(datasets_root / catalog.loc[(r.clip_id, r.frame_idx), "path"]))
        box = np.array([r.x1, r.y1, r.x2, r.y2])
        x0, y0, side = crop_window(box, r.zoom_scale, context_px, img.shape[1], img.shape[0])
        tile = cv2.resize(img[y0:y0 + side, x0:x0 + side], (thumb, thumb), interpolation=cv2.INTER_CUBIC)
        f = thumb / side
        color = (60, 170, 60) if r.zoom_new_votes > 0 else (40, 40, 230)
        cv2.rectangle(tile, (int((r.x1 - x0) * f), int((r.y1 - y0) * f)), (int((r.x2 - x0) * f), int((r.y2 - y0) * f)), color, 2)
        cv2.putText(tile, f"k{r.k}->{r.k_zoom} {r.dist_m:.0f}m", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if r.zoom_decoy:
            cv2.putText(tile, r.zoom_decoy[:20], (5, thumb - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
        row, col = divmod(i, cols)
        sheet[row * thumb:(row + 1) * thumb, col * thumb:(col + 1) * thumb] = tile
    cv2.imwrite(str(stage.viz_dir / "crops.jpg"), sheet)
    LOGGER.info("Viz: %d crops -> %s", n, stage.viz_dir / "crops.jpg")


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    parser = base_arg_parser(__doc__)
    parser.add_argument("--fused-path", type=Path, default=None, help="Stage 03 fused.parquet (default: from config).")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    stage = start_stage("03b_zoom", cfg, args)
    zcfg, fcfg, pcfg = cfg["zoom"], cfg["fuse"], cfg["predict"]
    camera, _ = load_clips(Path(cfg["paths"]["clips"]))
    ref_len = float(camera["ref_length_m"])
    datasets_root = Path(cfg["paths"]["datasets"])
    catalog = pd.read_parquet(cfg["paths"]["catalog"]).set_index(["clip_id", "frame_idx"])

    fused = pd.read_parquet(args.fused_path or Path(zcfg["fused_path"]))
    mask, scale = zoom_candidates(fused, zcfg)
    fused["zoom_checked"], fused["zoom_scale"] = mask, scale
    cand = fused[mask]
    LOGGER.info("Zoom candidates: %d of %d boxes (k<=%s, upscale >= %sx)", len(cand), len(fused),
                zcfg["max_k"], zcfg["min_scale"])
    groups = list(cand.groupby(["clip_id", "frame_idx"], sort=True))
    paths = [datasets_root / catalog.loc[key, "path"] for key, _ in groups]

    fused["zoom_decoy"] = ""
    for name in zcfg["models"]:
        mcfg = dict(pcfg["models"][name])
        if mcfg["type"] == "grounding_dino":  # decoys compete with "car" for the box; keep their labels
            mcfg.update(decoys=list(zcfg.get("decoys", [])), keep_decoys=True)
        det = build_detector(name, mcfg, pcfg["device"])
        targets = getattr(det, "targets", None)
        col = f"zoom_conf_{name}"
        fused[col] = np.nan
        t0 = time.perf_counter()
        for ((clip_id, frame_idx), g), img in zip(groups, iter_images(paths)):
            conf, labels = zoom_frame(
                det, img, g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64), g["zoom_scale"].to_numpy(),
                float(mcfg["vote_conf"]), float(catalog.loc[(clip_id, frame_idx), "f_px"]), fcfg, ref_len)
            vehicle = np.array([targets is None or is_target(lbl, targets) for lbl in labels])
            fused.loc[g.index, col] = np.where(vehicle, conf, np.nan)
            decoy = [lbl if (not v and np.isfinite(c)) else "" for lbl, v, c in zip(labels, vehicle, conf)]
            fused.loc[g.index, "zoom_decoy"] = np.where(fused.loc[g.index, "zoom_decoy"] != "",
                                                        fused.loc[g.index, "zoom_decoy"], decoy)
        LOGGER.info("%s: %d boxes in %d frames, %.1f s", name, len(cand), len(groups), time.perf_counter() - t0)
        del det

    conf_cols = [f"zoom_conf_{m}" for m in zcfg["models"]]
    fused["zoom_votes"] = fused[conf_cols].notna().sum(axis=1).astype(int)
    voted = fused["models"].str.split("+")
    fused["zoom_new_votes"] = sum(
        (fused[f"zoom_conf_{m}"].notna() & ~voted.apply(lambda ms, m=m: m in ms)).astype(int) for m in zcfg["models"])
    fused["k_zoom"] = fused["k"] + fused["zoom_new_votes"]
    fused.to_parquet(stage.out_dir / "fused_zoom.parquet", index=False)
    checked = fused[fused["zoom_checked"]]
    stats = compute_stats(checked, list(zcfg["models"]))
    LOGGER.info("Gained a vote by zoom: %s", {k: v["gained_vote"] for k, v in stats["by_k"].items()})
    LOGGER.info("Decoy label at zoom: %s", {k: v["decoy"] for k, v in stats["by_k"].items()})
    stage.save_stats(stats)
    render_viz(stage, checked, catalog, datasets_root, zcfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
