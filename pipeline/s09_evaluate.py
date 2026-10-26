"""Stage 09: run trained detectors on whole frames and score them per distance band.

Inference is the same slicing as the auto-labeller (640 tiles at native resolution, 20% overlap,
cut boxes dropped, one full-frame pass, NMS). Every frame of a clip is evaluated at its native
frame rate; labels made on every step-th frame are interpolated per vehicle in between (stage 06).

Splits:
    val   the val clip with the auto labels (tiers A-C, double boxes removed). Model selection and
          the confidence threshold are decided here.
    eval  the eval clips with the reviewed ground truth (stage 06), run once for the final numbers,
          with the threshold taken from the val run of the same model (never tuned on eval).

Metrics (bands by distance from the box size, pinhole model; a missed box uses its label's
distance, a false alarm its own):
    detection rate   matched label boxes / label boxes (match greedy by conf, the auto-labeller's rule:
                     IoU >= ``match_iou`` for pairs nearer than ``fuse.near_max_m``, NWD >= ``match_nwd``
                     farther; ``iou_only_detection_rate`` in metrics.json is the same with IoU only)
    precision        matched detections / detections
    F1               2 P R / (P + R); ``F1_macro`` = mean over ``select_bands``
    FA/min           the task's formula: false alarms x 60 / N_frames (= per 60 frames); the
                     per-minute-of-video value (frames / native fps) is ``FA/min_real``
    TTFD             per vehicle: first labelled frame -> first frame it is detected, seconds
                     (mean and p90 over detected vehicles; the rest is ``never_detected``);
                     a vehicle belongs to the band where it first appears
    mAP@0.5          all-point interpolated AP at IoU 0.5 (VOC2010 style; bonus, threshold-free)

Selection (val): DR, precision, FA/min and TTFD all depend on the confidence threshold, so a model
is judged at its operating point, not by mAP. For each run the threshold maximises ``F1_macro``
(bands weighted equally: near vehicles are the majority and would hide the far band); the run
with the best ``F1_macro`` wins, within ``select_tie`` the better 200-400 m F1 decides. Written to
``val/selected.json``; ``--runs best`` on eval reads it.

Auto-labelling stages (eval, ``--runs stages``): the boxes of every stage are scored like a detector
against the reviewed ground truth, at the stage's own operating point:
    raw_<model>   one labeller's raw boxes, threshold = its vote_conf (stage 02)
                  (stages 02-03b ran on every step-th frame of a clip, so they are scored on those
                  frames only; the auto labels are interpolated to every frame like the ground truth)
    fused_all     the three fused, every box (k >= 1, stage 03)
    fused_k2      fused boxes at least two models agree on
    zoom_k2       k >= 2 after the zoom re-check of small doubtful boxes (stage 03b)
    autolabel     kept tracks (tiers A-C, stages 04-05), confidence = track score, no threshold:
                  the auto labels the detector was trained on (teacher vs student)
Caveat: the eval ground truth was made by reviewing these very tracks, so the comparison favours
the auto-labeller (its boxes are the ground truth wherever the reviewer kept them).

95% intervals (bootstrap, ``N_BOOT`` resamples, seed from the config): detection rate and TTFD
resample vehicles (boxes of one vehicle are not independent), precision and FA/min resample
1-second blocks of video (false alarms belong to no vehicle).

Outputs (outputs/09_eval/<split>/):
    <run>/preds.parquet   all detections (conf >= evaluate.conf); reused only while preds.json (weights hash +
                          slicing settings) still matches, or re-run with --overwrite
    <run>/metrics.json    metrics per band and per clip, threshold, PR curve (val), error breakdown
    <run>/viz/story_*     eval: auto labels | my review (kept / added / removed) | the model, side by side
    <run>/viz/            whole-clip videos and stills, GT and detections overlaid (eval: clips from the
                          config; val: every val clip, for diagnosis); crop sheets labels_by_hit_<run>.jpg
                          (labelled vehicles by how often they are found) and false_alarms_<run>.jpg
    metrics.csv           one row per run, with 95% intervals (``*_lo`` / ``*_hi``)
    table_stages.md       eval: the stages and the trained runs side by side (README table)
    table_<run>.md        the task's table (metric x band) for each run
    selected.json         val only: the chosen run and its threshold

Example:
    python -m pipeline.s09_evaluate --runs n_step3 n_step6 n_step9
    python -m pipeline.s09_evaluate --runs best stages --split eval
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from pipeline.common import (VideoSink, base_arg_parser, frame_steps, load_clips, load_yaml, start_stage,
                             vehicle_bands)
from pipeline.detectors import build_detector
from pipeline.geometry import iou_one_to_many, nwd_one_to_many
from pipeline.s02_predict import iter_images, predict_frame
from pipeline.s03_fuse import band_labels
from pipeline.s06_import import interpolate_frames
from pipeline.s07_dataset import label_boxes
from pipeline.viz import crop_sheet, draw_boxes, render_boxes_video

LOGGER = logging.getLogger("pipeline.s09_evaluate")
BANDS = ["0-200", "200-400", ">400"]
AUTOLABEL = "autolabel"  # pseudo-run: the auto labels themselves (teacher) scored like a detector
BEST = "best"            # eval: the run chosen on val (val/selected.json)
STAGES_ALIAS = "stages"  # eval: every auto-labelling stage below, in pipeline order
STAGES = {  # pseudo-run -> table name; boxes of each auto-labelling stage scored like a detector (eval only)
    "raw_yolo26x": "YOLO26x alone",
    "raw_rtdetr_x": "RT-DETR-x alone",
    "raw_gdino_base": "Grounding DINO alone",
    "fused_all": "3 models fused, all boxes (k>=1)",
    "fused_k2": "fused, >= 2 models agree (k>=2)",
    "zoom_k2": "+ zoom re-check, k>=2",
    AUTOLABEL: "+ tracks and triage = auto labels",
}
N_BOOT = 1000            # bootstrap resamples of the 95% intervals
STAGE_FRAMES = "@stage_frames"  # suffix: a trained run scored only on the frames the auto-labelling stages saw
THR_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
OVERLAY = {  # kind -> (BGR colour, thickness, legend text); drawn in this order (GT under detections)
    "gt_hit": ((0, 200, 0), 3, "GT, detected"),
    "gt_miss": ((0, 140, 255), 2, "GT, missed (distance)"),
    "tp": ((255, 200, 0), 1, "detection on GT"),
    "fp": ((0, 0, 255), 2, "false alarm (conf)"),
}


def eval_frames(catalog: pd.DataFrame, clips: list[Any], role: str) -> pd.DataFrame:
    """Every frame of the split's clips (metrics are per frame, at the clip's native frame rate).

    Args:
        catalog: Frame catalog (stage 01 extracts every frame).
        clips: Clip registry.
        role: ``val`` or ``eval``.

    Returns:
        Frames with ``step`` = 1.
    """
    ids = [c.clip_id for c in clips if c.role == role]
    frames = catalog[catalog["clip_id"].isin(ids)].sort_values(["clip_id", "frame_idx"], ignore_index=True)
    return frames.assign(step=1)


def predict_split(weights: Path, frames: pd.DataFrame, cfg: dict[str, Any], datasets_root: Path) -> pd.DataFrame:
    """Sliced inference of one model on all frames.

    Args:
        weights: ``best.pt``.
        frames: Frames to run on.
        cfg: Full pipeline config (``evaluate`` and ``predict`` sections).
        datasets_root: Root of the frame paths.

    Returns:
        Detections: ``clip_id, frame_idx, x1..y2, conf, pass`` (``tile`` / ``full``: which pass survived NMS).
    """
    ecfg = cfg["evaluate"]
    det = build_detector("trained", {"type": "yolo", "weights": str(weights), "tile": int(ecfg["tile"]),
                                     "batch": int(ecfg["batch"]), "conf": float(ecfg["conf"]), "classes": [0],
                                     "quantize": int(ecfg.get("quantize", 16))}, cfg["predict"]["device"])
    slicing = {k: ecfg[k] for k in ("tile_overlap", "edge_margin_px", "full_frame_pass", "nms_iou")}
    parts = []
    for i, (fr, img) in enumerate(zip(frames.itertuples(index=False),
                                      iter_images([datasets_root / p for p in frames["path"]]))):
        out = predict_frame(det, img, slicing)
        parts.append(out[["x1", "y1", "x2", "y2", "conf", "pass"]].assign(clip_id=fr.clip_id, frame_idx=fr.frame_idx))
        if (i + 1) % 200 == 0:
            LOGGER.info("  %d / %d frames", i + 1, len(frames))
    cols = ["clip_id", "frame_idx", "x1", "y1", "x2", "y2", "conf", "pass"]
    return pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)


def cached_predictions(run_dir: Path, weights: Path, frames: pd.DataFrame, cfg: dict[str, Any], datasets_root: Path,
                       overwrite: bool) -> pd.DataFrame:
    """Detections of a trained run, cached in ``preds.parquet`` and re-run when the weights or settings change.

    ``preds.json`` next to the cache holds the weights' SHA-1, the number of frames and the slicing
    settings; a mismatch (retrained model, other config) re-runs inference instead of silently
    scoring old detections.

    Args:
        run_dir: Result folder of the run.
        weights: ``best.pt`` of the run.
        frames: Frames to run on.
        cfg: Full pipeline config.
        datasets_root: Root of the frame paths.
        overwrite: Re-run inference regardless of the cache.

    Returns:
        Detections: ``clip_id, frame_idx, x1..y2, conf, pass``.
    """
    if not weights.is_file():
        raise FileNotFoundError(f"{weights}: train the run first (stage 08)")
    ecfg = cfg["evaluate"]
    key = {"weights": str(weights), "sha1": hashlib.sha1(weights.read_bytes()).hexdigest(), "frames": len(frames),
           **{k: ecfg[k] for k in ("tile", "tile_overlap", "edge_margin_px", "full_frame_pass", "nms_iou", "conf",
                                   "quantize")}}
    pred_path, key_path = run_dir / "preds.parquet", run_dir / "preds.json"
    if (not overwrite and pred_path.is_file() and key_path.is_file()
            and json.loads(key_path.read_text(encoding="utf-8")) == key):
        return pd.read_parquet(pred_path)
    if pred_path.is_file() and not overwrite:
        LOGGER.info("%s: cached detections are from other weights or settings, re-running", run_dir.name)
    LOGGER.info("%s: inference on %d frames", run_dir.name, len(frames))
    preds = predict_split(weights, frames, cfg, datasets_root)
    preds.to_parquet(pred_path, index=False)
    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    return preds


def val_threshold(path: Path) -> float:
    """Confidence threshold the run got on val (eval never tunes it).

    Args:
        path: ``09_eval/val/<run>/metrics.json``.

    Returns:
        Threshold.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path}: evaluate the run on val first (the threshold comes from val)")
    return float(json.loads(path.read_text(encoding="utf-8"))["threshold"])


def autolabel_predictions(cfg: dict[str, Any], clips: list[Any], clip_ids: set[str]) -> pd.DataFrame:
    """The auto-labeller's own output as "detections": kept tracks (tiers A-C), one box per vehicle
    and frame, interpolated to every frame; confidence = track score (stage 04c / 05).

    Args:
        cfg: Full pipeline config.
        clips: Clip registry.
        clip_ids: Clips to return.

    Returns:
        ``clip_id, frame_idx, x1..y2, conf``.
    """
    boxes = pd.read_parquet(cfg["dataset"]["boxes_path"])
    tiers = pd.read_parquet(cfg["dataset"]["tiers_path"])
    lab = label_boxes(boxes[boxes["clip_id"].isin(clip_ids)], tiers, cfg["dataset"], cfg["import"])
    steps = frame_steps(clips, int(cfg["predict"]["train_frame_step"]))
    lab = pd.concat([interpolate_frames(g, int(steps[c])) for c, g in lab.groupby("clip_id")], ignore_index=True)
    score = tiers.set_index(["clip_id", "track_id"])["score"]
    lab["conf"] = score.reindex(pd.MultiIndex.from_frame(lab[["clip_id", "track_id"]])).to_numpy()
    return lab[["clip_id", "frame_idx", "x1", "y1", "x2", "y2", "conf"]].fillna({"conf": 0.0})


def match(preds: pd.DataFrame, labels: pd.DataFrame, iou_thr: float, nwd: dict[str, float] | None = None
          ) -> tuple[np.ndarray, np.ndarray]:
    """Greedy matching per frame in descending confidence (valid for every threshold cut).

    With ``nwd`` the rule is the auto-labeller's (stage 03): a pair whose mean distance is below
    ``near_max_m`` needs IoU >= ``iou_thr``, a farther pair needs NWD >= ``thr`` (IoU is brittle on
    15 px boxes). Without it every pair needs IoU >= ``iou_thr`` (mAP).

    Args:
        preds: Detections, index 0..N-1 (``dist_m`` needed with ``nwd``).
        labels: Label boxes, index 0..M-1 (``dist_m`` needed with ``nwd``).
        iou_thr: Minimum IoU.
        nwd: ``{"thr", "c", "near_max_m"}`` or None.

    Returns:
        Per detection: matched label row (-1 = false alarm); per label: confidence of the
        detection that matched it (-1 = never matched).
    """
    p_match = np.full(len(preds), -1)
    l_conf = np.full(len(labels), -1.0)
    pb, pc = preds[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float), preds["conf"].to_numpy(dtype=float)
    lb = labels[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    if nwd:
        pdist, ldist = preds["dist_m"].to_numpy(dtype=float), labels["dist_m"].to_numpy(dtype=float)
    lab_rows = labels.groupby(["clip_id", "frame_idx"]).indices
    for key, p_rows in preds.groupby(["clip_id", "frame_idx"]).indices.items():
        l_rows = lab_rows.get(key)
        if l_rows is None:
            continue
        free = np.ones(len(l_rows), dtype=bool)
        for pi in p_rows[np.argsort(-pc[p_rows], kind="stable")]:
            iou = iou_one_to_many(pb[pi], lb[l_rows])
            if nwd:
                near = (pdist[pi] + ldist[l_rows]) / 2 < nwd["near_max_m"]
                sim = nwd_one_to_many(pb[pi], lb[l_rows], nwd["c"])
                score = np.where(near, iou, sim)
                ok = np.where(near, iou >= iou_thr, sim >= nwd["thr"])
            else:
                score, ok = iou, iou >= iou_thr
            score = np.where(free & ok, score, -1.0)
            j = int(np.argmax(score))
            if score[j] >= 0:
                free[j] = False
                p_match[pi], l_conf[l_rows[j]] = l_rows[j], pc[pi]
                if not free.any():
                    break
    return p_match, l_conf


def average_precision(conf: np.ndarray, tp: np.ndarray, n_labels: int) -> float:
    """All-point interpolated AP (COCO / VOC2010 style).

    Args:
        conf: Detection confidences.
        tp: True-positive flag per detection.
        n_labels: Number of label boxes.

    Returns:
        AP in [0, 1] (NaN without labels).
    """
    if n_labels == 0:
        return float("nan")
    order = np.argsort(-conf, kind="stable")
    tps = np.cumsum(tp[order])
    recall = tps / n_labels
    precision = tps / np.arange(1, len(order) + 1)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def band_of(matched: np.ndarray, own: np.ndarray, label_band: np.ndarray) -> np.ndarray:
    """Band of each detection: its label's band if matched, else from its own size.

    Args:
        matched: Matched label row per detection (-1 = none).
        own: Band from the detection's own size.
        label_band: Band per label row.

    Returns:
        Band per detection.
    """
    if not len(label_band):
        return own
    return np.where(matched >= 0, label_band[np.clip(matched, 0, len(label_band) - 1)], own)


def f1_score(p: Any, r: Any) -> Any:
    """Harmonic mean of precision and recall (0 where both are 0); scalars or arrays."""
    p, r = np.asarray(p, float), np.asarray(r, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(p + r > 0, 2 * p * r / (p + r), 0.0)


def macro_bands(labels: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    """Bands that enter ``F1_macro``: ``select_bands`` with at least ``select_min_boxes`` label boxes.

    Args:
        labels: Label boxes with ``band``.
        cfg: ``evaluate`` config section.

    Returns:
        Band names.
    """
    counts = labels["band"].value_counts()
    return [b for b in cfg["select_bands"] if counts.get(b, 0) >= int(cfg["select_min_boxes"])]


def threshold_curve(preds: pd.DataFrame, labels: pd.DataFrame, bands: list[str]) -> pd.DataFrame:
    """Precision, recall and F1 per band at every threshold of ``THR_GRID``; F1 macro over ``bands``.

    Args:
        preds: Detections with ``conf, band, m_op``.
        labels: Label boxes with ``band, c_op``.
        bands: Bands averaged into ``f1_macro``.

    Returns:
        One row per threshold.
    """
    def count_ge(values: np.ndarray) -> np.ndarray:
        s = np.sort(values)
        return len(s) - np.searchsorted(s, THR_GRID, side="left")

    conf, tp = preds["conf"].to_numpy(float), preds["m_op"].to_numpy() >= 0
    p_band = band_of(preds["m_op"].to_numpy(), preds["band"].to_numpy(), labels["band"].to_numpy())
    c_op, l_band = labels["c_op"].to_numpy(float), labels["band"].to_numpy()
    out = pd.DataFrame({"thr": THR_GRID})
    for band in ["all", *bands]:
        pm = np.ones(len(preds), bool) if band == "all" else p_band == band
        lm = np.ones(len(labels), bool) if band == "all" else l_band == band
        n_det, n_tp, n_lab = count_ge(conf[pm]), count_ge(conf[pm & tp]), int(lm.sum())
        p = np.divide(n_tp, n_det, out=np.zeros(len(THR_GRID)), where=n_det > 0)
        r = count_ge(c_op[lm]) / n_lab if n_lab else np.zeros(len(THR_GRID))
        out[f"precision_{band}"], out[f"recall_{band}"], out[f"f1_{band}"] = p, r, f1_score(p, r)
    out["f1_macro"] = out[[f"f1_{b}" for b in bands]].mean(axis=1) if bands else out["f1_all"]
    return out


def select_run(table: pd.DataFrame, tie: float) -> pd.Series:
    """Best run on val: highest ``F1_macro``; among runs within ``tie`` of it, the best 200-400 m F1.

    Args:
        table: val metrics, one row per run.
        tie: F1 difference treated as a tie.

    Returns:
        The chosen row.
    """
    cand = table[~table["run"].isin(list(STAGES))].dropna(subset=["F1_macro"])
    near = cand[cand["F1_macro"] >= cand["F1_macro"].max() - tie]
    key = "F1_200-400" if near["F1_200-400"].notna().any() else "F1_macro"
    return near.sort_values([key, "F1_macro"], ascending=False, na_position="last").iloc[0]


def ttfd_per_vehicle(labels: pd.DataFrame, thr: float) -> pd.DataFrame:
    """Time to first detection of every labelled vehicle.

    Args:
        labels: Label boxes with ``clip_id, vehicle_id, frame_idx, t_s, band, c_op``.
        thr: Confidence threshold.

    Returns:
        Per vehicle: ``band`` (where it first appears, ``common.vehicle_bands``) and ``ttfd`` =
        first detected frame - first labelled frame, seconds (NaN = never detected).
    """
    keys = ["clip_id", "vehicle_id"]
    t0 = labels.groupby(keys)["t_s"].min()
    hit = labels[labels["c_op"] >= thr].groupby(keys)["t_s"].min()
    return pd.DataFrame({"band": vehicle_bands(labels, keys), "ttfd": hit.reindex(t0.index) - t0})


def score(preds: pd.DataFrame, labels: pd.DataFrame, frames: pd.DataFrame, thr: float, cfg: dict[str, Any]
          ) -> dict[str, Any]:
    """All metrics at one confidence threshold, per band and overall.

    Args:
        preds: Detections with ``band`` (own size), ``m_op`` (matched label row at match_iou), ``m_ap``.
        labels: Label boxes with ``band``, ``vehicle_id``, ``t_s``, ``c_op`` (conf of the matching detection).
        frames: Evaluated frames (``clip_id, frame_idx, t_s, step, fps``).
        thr: Confidence threshold.
        cfg: ``evaluate`` config section.

    Returns:
        Metrics dictionary.
    """
    minutes = float((frames.groupby("clip_id").size() / (frames.groupby("clip_id")["fps"].first()
                                                         / frames.groupby("clip_id")["step"].first()) / 60).sum())
    kept = preds[preds["conf"] >= thr]
    tp = kept["m_op"] >= 0
    lab_band = labels["band"].to_numpy()
    p_band = band_of(kept["m_op"].to_numpy(), kept["band"].to_numpy(), lab_band)
    detected = labels["c_op"] >= thr
    out: dict[str, Any] = {"threshold": round(thr, 3), "minutes": round(minutes, 2), "frames": len(frames)}
    for band in ["all", *BANDS]:
        lm = np.ones(len(labels), bool) if band == "all" else (labels["band"] == band).to_numpy()
        pm = np.ones(len(kept), bool) if band == "all" else (p_band == band)
        n_lab, n_det = int(lm.sum()), int(pm.sum())
        fp = int((pm & ~tp.to_numpy()).sum())
        dr = float(detected[lm].mean()) if n_lab else None
        prec = float(tp.to_numpy()[pm].mean()) if n_det else None
        b: dict[str, Any] = {"label_boxes": n_lab, "detections": n_det,
                             "detection_rate": round(dr, 4) if dr is not None else None,
                             "precision": round(prec, 4) if prec is not None else None,
                             "f1": round(float(f1_score(prec or 0.0, dr)), 4) if n_lab else None,
                             "false_alarms": fp,
                             "fa_per_min": round(fp * 60 / len(frames), 2) if len(frames) else None,  # task formula
                             "fa_per_min_real": round(fp / minutes, 2) if minutes else None}
        ap_mask = (np.ones(len(preds), bool) if band == "all" else
                   band_of(preds["m_ap"].to_numpy(), preds["band"].to_numpy(), lab_band) == band)
        b["map50"] = round(average_precision(preds["conf"].to_numpy()[ap_mask],
                                             (preds["m_ap"].to_numpy() >= 0)[ap_mask], n_lab), 4) if n_lab else None
        out[band] = b
    used = macro_bands(labels, cfg)
    out["f1_macro_bands"] = used
    out["f1_macro"] = round(float(np.mean([out[b]["f1"] for b in used])), 4) if used else out["all"]["f1"]
    # time to first detection, per vehicle (band where it first appears)
    first = ttfd_per_vehicle(labels, thr) if len(labels) else pd.DataFrame(columns=["band", "ttfd"])
    for band in ["all", *BANDS]:
        v = first if band == "all" else first[first["band"] == band]
        if len(v):
            t = v["ttfd"].dropna()
            out[band]["vehicles"] = len(v)
            out[band]["ttfd_s"] = {"mean": round(float(t.mean()), 2) if len(t) else None,
                                   "median": round(float(t.median()), 2) if len(t) else None,
                                   "p90": round(float(t.quantile(0.9)), 2) if len(t) else None,
                                   "never_detected": round(float(v["ttfd"].isna().mean()), 3)}
    out["per_clip"] = {}
    for c, g in frames.groupby("clip_id", sort=True):
        lc = (labels["clip_id"] == c).to_numpy()
        fp = int(((kept["clip_id"] == c) & ~tp).sum())
        out["per_clip"][c] = {"frames": len(g), "vehicles": int(labels.loc[lc, "vehicle_id"].nunique()),
                              "label_boxes": int(lc.sum()),
                              "detection_rate": round(float(detected[lc].mean()), 4) if lc.any() else None,
                              "false_alarms": fp, "fa_per_min": round(fp * 60 / len(g), 2),
                              "fa_per_min_real": round(fp / (len(g) / (g["fps"].iloc[0] / g["step"].iloc[0]) / 60), 2)}
    return out


def max_iou(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    """Per row of ``a``: the highest IoU with any box of ``b`` in the same frame (0 if none).

    Args:
        a: Boxes (``clip_id, frame_idx, x1..y2``).
        b: Boxes (same columns).

    Returns:
        IoU per row of ``a``, in row order.
    """
    out = np.zeros(len(a))
    ab, bb = a[["x1", "y1", "x2", "y2"]].to_numpy(float), b[["x1", "y1", "x2", "y2"]].to_numpy(float)
    b_rows = b.groupby(["clip_id", "frame_idx"]).indices
    for key, rows in a.groupby(["clip_id", "frame_idx"]).indices.items():
        br = b_rows.get(key)
        if br is not None:
            for i in rows:
                out[i] = iou_one_to_many(ab[i], bb[br]).max()
    return out


def error_breakdown(preds: pd.DataFrame, labels: pd.DataFrame, thr: float, touch_iou: float = 0.1) -> pd.DataFrame:
    """Where the errors come from at the operating point, per band.

    Detections: ``tp``; ``fp_overlap`` (touches a label: duplicate, part of a vehicle, loose box);
    ``fp_background`` (nothing labelled there), split by inference pass (``tile`` / ``full``).
    Labels: ``hit``; ``miss_overlap`` (a detection touches it, but not well enough or it is taken);
    ``miss_none`` (no detection there).

    Args:
        preds: Detections with ``conf, band, m_op`` (``pass`` optional).
        labels: Label boxes with ``band, c_op``.
        thr: Confidence threshold.
        touch_iou: IoU from which two boxes "touch".

    Returns:
        Counts: ``what, pass, outcome, band, n``.
    """
    kept = preds[preds["conf"] >= thr].reset_index(drop=True)
    p_band = band_of(kept["m_op"].to_numpy(), kept["band"].to_numpy(), labels["band"].to_numpy())
    touch = max_iou(kept, labels) >= touch_iou
    det = pd.DataFrame({"what": "detection", "pass": kept["pass"] if "pass" in kept else "?", "band": p_band,
                        "outcome": np.where(kept["m_op"] >= 0, "tp", np.where(touch, "fp_overlap", "fp_background"))})
    touch = max_iou(labels, kept) >= touch_iou
    lab = pd.DataFrame({"what": "label", "pass": "-", "band": labels["band"].to_numpy(),
                        "outcome": np.where(labels["c_op"] >= thr, "hit", np.where(touch, "miss_overlap", "miss_none"))})
    both = pd.concat([det, lab], ignore_index=True)
    return both.groupby(["what", "pass", "outcome", "band"]).size().rename("n").reset_index()


def overlay_boxes(preds: pd.DataFrame, labels: pd.DataFrame, thr: float) -> pd.DataFrame:
    """GT and kept detections in one table for drawing: ``kind`` (see ``OVERLAY``) and a short label.

    Args:
        preds: Detections with ``conf, m_op``.
        labels: Label boxes with ``c_op, dist_m``.
        thr: Confidence threshold (the operating point).

    False alarms are labelled with their confidence, `` F`` = from the full-frame pass.

    Returns:
        ``clip_id, frame_idx, x1..y2, kind, text``, GT rows first (drawn under the detections).
    """
    gt = labels.assign(kind=np.where(labels["c_op"] >= thr, "gt_hit", "gt_miss"))
    gt["text"] = np.where(gt["kind"] == "gt_miss", gt["dist_m"].round().astype(int).astype(str) + "m", "")
    kept = preds[preds["conf"] >= thr]
    det = kept.assign(kind=np.where(kept["m_op"] >= 0, "tp", "fp"))
    full = det["pass"].eq("full").to_numpy() if "pass" in det else np.zeros(len(det), bool)
    det["text"] = np.where(det["kind"] == "fp", det["conf"].map("{:.2f}".format) + np.where(full, " F", ""), "")
    cols = ["clip_id", "frame_idx", "x1", "y1", "x2", "y2", "kind", "text"]
    return pd.concat([gt[cols], det[cols]], ignore_index=True)


def render_eval_viz(out_dir: Path, overlay: pd.DataFrame, frames: pd.DataFrame, run: str, thr: float,
                    vcfg: dict[str, Any], datasets_root: Path) -> None:
    """Whole-clip videos and stills of the detector at its operating point with the GT overlaid.

    Stills are evenly spaced over the frames that have GT (not picked by hand).

    Args:
        out_dir: Output folder.
        overlay: ``overlay_boxes`` output.
        frames: Evaluated frames (``clip_id, frame_idx, t_s, path``).
        run: Run name (legend).
        thr: Confidence threshold (legend).
        vcfg: ``evaluate.viz`` config section.
        datasets_root: Root for catalog paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    legend = [(f"{run}, conf >= {thr:.2f}", (255, 255, 255))] + [(t, c) for c, _, t in OVERLAY.values()]

    def style(r: Any) -> tuple:
        return OVERLAY[r.kind][0], r.text, OVERLAY[r.kind][1]

    by_clip = {c: g for c, g in overlay.groupby("clip_id")}
    present = set(frames["clip_id"])
    for clip in map(str, vcfg.get("video_clips", [])):
        if clip in present:
            render_boxes_video(out_dir / f"{clip}_{run}.mp4", frames[frames["clip_id"] == clip],
                               by_clip.get(clip, overlay.iloc[:0]), datasets_root, 0, style, legend)
            LOGGER.info("  video: %s", out_dir / f"{clip}_{run}.mp4")
    n = int(vcfg.get("examples_per_clip", 3))
    for clip in map(str, vcfg.get("example_clips", [])):
        boxes = by_clip.get(clip)
        if clip not in present or boxes is None:
            continue
        with_gt = np.sort(boxes.loc[boxes["kind"].str.startswith("gt"), "frame_idx"].unique())
        if not len(with_gt):
            continue
        fr = frames[frames["clip_id"] == clip].set_index("frame_idx")
        for idx in dict.fromkeys(with_gt[int((i + 0.5) / n * len(with_gt))] for i in range(n)):
            img = draw_boxes(datasets_root / fr.at[idx, "path"], boxes[boxes["frame_idx"] == idx], style,
                             legend + [(f"clip {clip}, t = {fr.at[idx, 't_s']:.1f} s", (255, 255, 255))])
            cv2.imwrite(str(out_dir / f"{clip}_{int(idx):06d}_{run}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    LOGGER.info("  stills: %s", out_dir)


def review_boxes(auto: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """What the manual review changed, per frame: auto boxes kept, boxes added by hand, auto boxes removed.

    An auto box counts as kept when a reviewed box of the same frame overlaps it (IoU >= 0.5, one to
    one); otherwise it was removed (wrong track, or a double box dropped on import).

    Args:
        auto: Auto labels (``autolabel_predictions``).
        gt: Reviewed ground truth with ``source`` (``manual`` = drawn by hand).

    Returns:
        ``clip_id, frame_idx, x1..y2, kind`` (``kept`` / ``added`` / ``removed``).
    """
    a = auto.reset_index(drop=True)
    m, _ = match(a, gt.reset_index(drop=True), 0.5)
    manual = gt["source"].astype(str).eq("manual") if "source" in gt.columns else pd.Series(False, index=gt.index)
    kept = gt.assign(kind=np.where(manual, "added", "kept"))
    cols = ["clip_id", "frame_idx", "x1", "y1", "x2", "y2", "kind"]
    return pd.concat([kept[cols], a[m < 0].assign(kind="removed")[cols]], ignore_index=True)


def render_story(out_dir: Path, frames: pd.DataFrame, auto: pd.DataFrame, review: pd.DataFrame,
                 overlay: pd.DataFrame, run: str, thr: float, vcfg: dict[str, Any], datasets_root: Path,
                 max_side: int = 1280) -> None:
    """The eval clip three times side by side: auto labels | my review | the trained model.

    Stills (``story_<clip>_<frame>.jpg``, the same evenly spaced frames as the other stills) for
    ``example_clips`` and whole-clip videos (``story_<clip>.mp4``) for ``video_clips``.

    Args:
        out_dir: Output folder.
        frames: Evaluated frames (``clip_id, frame_idx, t_s, path``).
        auto: Auto labels.
        review: ``review_boxes`` output.
        overlay: ``overlay_boxes`` output of the model.
        run: Run name (legend).
        thr: Confidence threshold (legend).
        vcfg: ``evaluate.viz`` config section.
        datasets_root: Root for catalog paths.
        max_side: Long side of each panel.
    """
    white = (255, 255, 255)
    panels = [
        (auto.assign(kind="auto"), {"auto": ((0, 220, 255), 2, "auto label (tiers A-C)")}, "1. auto labels"),
        (review, {"kept": ((0, 200, 0), 2, "kept"), "added": ((255, 200, 0), 2, "added by hand"),
                  "removed": ((0, 0, 255), 2, "removed by hand")}, "2. after my review"),
        (overlay, {k: (c, t, txt) for k, (c, t, txt) in OVERLAY.items()}, f"3. model {run}, conf >= {thr:.2f}"),
    ]
    split = [{c: g for c, g in boxes.groupby("clip_id")} for boxes, _, _ in panels]

    def frame_image(clip: str, fr: Any) -> np.ndarray:
        parts = []
        for (boxes, kinds, title), by_clip in zip(panels, split):
            cb = by_clip.get(clip)
            sel = cb[cb["frame_idx"] == fr.frame_idx] if cb is not None else None
            text = boxes["text"] if "text" in boxes.columns else None

            def style(r: Any, kinds: dict = kinds, text: Any = text) -> tuple:
                return kinds[r.kind][0], (r.text if text is not None else ""), kinds[r.kind][1]

            legend = [(title, white)] + [(t, c) for c, _, t in kinds.values()]
            parts.append(draw_boxes(datasets_root / fr.path, sel, style, legend, max_side))
        gap = np.full((parts[0].shape[0], 8, 3), 255, np.uint8)
        return np.hstack([parts[0], gap, parts[1], gap, parts[2]])

    out_dir.mkdir(parents=True, exist_ok=True)
    present = set(frames["clip_id"])
    for clip in map(str, vcfg.get("video_clips", [])):
        fr_c = frames[frames["clip_id"] == clip]
        if clip not in present or len(fr_c) < 2:
            continue
        fps = max(1.0, (len(fr_c) - 1) / max(fr_c["t_s"].iloc[-1] - fr_c["t_s"].iloc[0], 1e-6))
        with VideoSink(out_dir / f"story_{clip}.mp4", fps) as sink:
            for fr in fr_c.itertuples(index=False):
                sink.write(frame_image(clip, fr))
        LOGGER.info("  story video: %s", out_dir / f"story_{clip}.mp4")
    n = int(vcfg.get("examples_per_clip", 3))
    for clip in map(str, vcfg.get("example_clips", [])):
        gt_frames = np.sort(review.loc[(review["clip_id"] == clip) & (review["kind"] != "removed"), "frame_idx"].unique())
        if clip not in present or not len(gt_frames):
            continue
        fr_c = frames[frames["clip_id"] == clip].set_index("frame_idx", drop=False)
        for idx in dict.fromkeys(gt_frames[int((i + 0.5) / n * len(gt_frames))] for i in range(n)):
            cv2.imwrite(str(out_dir / f"story_{clip}_{int(idx):06d}.jpg"), frame_image(clip, fr_c.loc[idx]),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])


def render_error_sheets(out_dir: Path, run: str, preds: pd.DataFrame, labels: pd.DataFrame, frames: pd.DataFrame,
                        thr: float, tiers: pd.DataFrame, datasets_root: Path, cols: int = 12, n_fp: int = 48) -> None:
    """Crop sheets that check the labels against the detector (and the other way round).

    ``labels_by_hit_<run>.jpg``: every labelled vehicle (its middle box, 3x context), rows by the share
    of its boxes the detector finds; caption = tier letter + vehicle id, share found, distance.
    Never-found "vehicles" that are not vehicles = wrong labels.
    ``false_alarms_<run>.jpg``: the most confident false alarms on empty places (no label touches them),
    one per 64 px cell of a clip; caption = conf, distance, ``F`` = full-frame pass.
    Vehicles here = vehicles missing from the labels.

    Args:
        out_dir: Output folder.
        run: Run name (file names).
        preds: Detections with ``conf, m_op, dist_m`` (``pass`` optional).
        labels: Label boxes with ``c_op, dist_m, vehicle_id`` (``track_id`` / ``tier`` optional).
        frames: Evaluated frames (``clip_id, frame_idx, path``).
        thr: Confidence threshold.
        tiers: Stage 05 track table, gives the tier of auto-label tracks.
        datasets_root: Root for catalog paths.
        cols: Crops per row.
        n_fp: False alarms shown.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = frames.set_index(["clip_id", "frame_idx"])["path"]

    def with_path(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(path=paths.reindex(pd.MultiIndex.from_frame(df[["clip_id", "frame_idx"]])).to_numpy())

    def rows_of(title: str, part: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
        return [(f"{title}\n{len(part)}" if i == 0 else "", part.iloc[i:i + cols]) for i in range(0, len(part), cols)]

    lab = labels.assign(hit=labels["c_op"] >= thr)
    if "tier" not in lab and "track_id" in lab:
        lab = lab.merge(tiers[["clip_id", "track_id", "tier"]], on=["clip_id", "track_id"], how="left")
    lab["tier"] = lab["tier"].fillna("?") if "tier" in lab else "?"
    per = (lab.groupby(["clip_id", "vehicle_id"])
           .agg(hit=("hit", "mean"), dist=("dist_m", "median"), tier=("tier", "first")).reset_index())
    per["track_id"] = per["vehicle_id"]
    groups = (rows_of("never found", per[per["hit"] == 0])
              + rows_of("found < 50%", per[(per["hit"] > 0) & (per["hit"] < 0.5)].sort_values("hit"))
              + rows_of("found >= 50%", per[per["hit"] >= 0.5].sort_values("hit")))
    crop_sheet(groups, with_path(lab.assign(track_id=lab["vehicle_id"])), datasets_root,
               out_dir / f"labels_by_hit_{run}.jpg", cols,
               lambda t: f"{str(t.tier)[:1]}{t.vehicle_id} {t.hit:.0%} {t.dist:.0f}m")

    fa = preds[(preds["conf"] >= thr) & (preds["m_op"] < 0)].reset_index(drop=True)
    fa = fa[max_iou(fa, labels) < 0.1]
    fa = fa.assign(cx=((fa["x1"] + fa["x2"]) / 2 // 64).astype(int), cy=((fa["y1"] + fa["y2"]) / 2 // 64).astype(int))
    fa = fa.sort_values("conf", ascending=False).drop_duplicates(["clip_id", "cx", "cy"]).head(n_fp)
    fa = with_path(fa.assign(track_id=np.arange(len(fa)), full=fa["pass"].eq("full") if "pass" in fa else False))
    crop_sheet(rows_of("false alarms,\nmost confident", fa), fa, datasets_root, out_dir / f"false_alarms_{run}.jpg",
               cols, lambda t: f"{t.conf:.2f} {t.dist_m:.0f}m{' F' if t.full else ''}")
    LOGGER.info("  sheets: %s, %s", out_dir / f"labels_by_hit_{run}.jpg", out_dir / f"false_alarms_{run}.jpg")


def stage_predictions(run: str, cfg: dict[str, Any], clips: list[Any], clip_ids: set[str]
                      ) -> tuple[pd.DataFrame, float]:
    """Boxes of one auto-labelling stage as detections, with the stage's operating threshold.

    Args:
        run: Key of ``STAGES``.
        cfg: Full pipeline config.
        clips: Clip registry.
        clip_ids: Clips to return.

    Returns:
        ``clip_id, frame_idx, x1..y2, conf`` (+ ``pass`` for raw boxes) and the threshold.
    """
    cols = ["clip_id", "frame_idx", "x1", "y1", "x2", "y2", "conf"]
    if run == AUTOLABEL:
        return autolabel_predictions(cfg, clips, clip_ids), 0.0
    if run.startswith("raw_"):
        model = run[len("raw_"):]
        df = pd.read_parquet(Path(cfg["fuse"]["predict_dir"]) / f"{model}.parquet")
        df = df[df["clip_id"].isin(clip_ids)]
        return df[cols + ["pass"]].reset_index(drop=True), float(cfg["predict"]["models"][model]["vote_conf"])
    if run == "zoom_k2":
        df = pd.read_parquet(cfg["track"]["input_path"])
        df = df[df["k_zoom"] >= 2]
    else:
        df = pd.read_parquet(cfg["zoom"]["fused_path"])
        df = df[df["k"] >= (2 if run == "fused_k2" else 1)]
    df = df[df["clip_id"].isin(clip_ids)]
    return df.assign(conf=df["conf_mean"])[cols].reset_index(drop=True), 0.0


def processed_frames(frames: pd.DataFrame, labels: pd.DataFrame, clips: list[Any], train_step: int
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frames (and their labels) that the auto-labelling stages 02-03b ran on: every step-th frame.

    Args:
        frames: All evaluated frames.
        labels: Label boxes of all frames.
        clips: Clip registry (``frame_step``).
        train_step: Default step of train / val clips.

    Returns:
        Frames with ``step`` set to the clip's step (time per frame for FA/min), and their labels.
    """
    steps = frame_steps(clips, train_step)
    fstep = frames["clip_id"].map(steps)
    on = (frames["frame_idx"] % fstep == 0).to_numpy()
    lab_on = (labels["frame_idx"] % labels["clip_id"].map(steps) == 0).to_numpy()
    return frames[on].assign(step=fstep[on]).reset_index(drop=True), labels[lab_on].reset_index(drop=True)


def bootstrap_ci(preds: pd.DataFrame, labels: pd.DataFrame, frames: pd.DataFrame, thr: float, bands: list[str],
                 seed: int, n_boot: int = N_BOOT) -> dict[str, dict[str, list[float | None]]]:
    """95% bootstrap intervals of detection rate, mean TTFD, precision and FA/min per band.

    Detection rate and TTFD resample vehicles; precision and FA/min resample 1-second blocks of
    video (per clip), because a false alarm belongs to no vehicle.

    Args:
        preds: Detections with ``conf, band, m_op, t_s``.
        labels: Label boxes with ``band, c_op, t_s, vehicle_id``.
        frames: Evaluated frames (``clip_id, t_s``).
        thr: Confidence threshold.
        bands: Bands to report (``all`` = every box).
        seed: Random seed.
        n_boot: Resamples.

    Returns:
        ``{band: {metric: [lo, hi]}}`` (None where the band is empty).
    """
    rng = np.random.default_rng(seed)

    def interval(values: np.ndarray) -> list[float | None]:
        values = values[np.isfinite(values)]
        return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)] \
            if len(values) else [None, None]

    kept = preds[preds["conf"] >= thr]
    p_band = band_of(kept["m_op"].to_numpy(), kept["band"].to_numpy(), labels["band"].to_numpy())
    det = pd.DataFrame({"clip_id": kept["clip_id"].to_numpy(), "block": np.floor(kept["t_s"].to_numpy()).astype(int),
                        "band": p_band, "tp": kept["m_op"].to_numpy() >= 0})
    blocks = frames.assign(block=np.floor(frames["t_s"]).astype(int)).groupby(["clip_id", "block"]).size()
    lab = labels.assign(hit=labels["c_op"] >= thr)
    first = ttfd_per_vehicle(labels, thr) if len(labels) else pd.DataFrame(columns=["band", "ttfd"])
    out: dict[str, dict[str, list[float | None]]] = {}
    for band in bands:
        lb = lab if band == "all" else lab[lab["band"] == band]
        per_v = lb.groupby(["clip_id", "vehicle_id"])["hit"].agg(["sum", "size"])
        res: dict[str, list[float | None]] = {"detection_rate": [None, None], "ttfd_mean_s": [None, None]}
        if len(per_v):
            idx = rng.integers(0, len(per_v), (n_boot, len(per_v)))
            h, n = per_v["sum"].to_numpy(float), per_v["size"].to_numpy(float)
            res["detection_rate"] = interval(h[idx].sum(1) / n[idx].sum(1))
        ttfd = (first if band == "all" else first[first["band"] == band])["ttfd"].to_numpy(float)
        if np.isfinite(ttfd).any():
            idx = rng.integers(0, len(ttfd), (n_boot, len(ttfd)))
            res["ttfd_mean_s"] = interval(np.array([np.nanmean(ttfd[i]) if np.isfinite(ttfd[i]).any() else np.nan
                                                    for i in idx]))
        db = det if band == "all" else det[det["band"] == band]
        g = db.groupby(["clip_id", "block"])["tp"].agg(["sum", "size"]).reindex(blocks.index, fill_value=0)
        idx = rng.integers(0, len(blocks), (n_boot, len(blocks)))
        tp, nd, nf = g["sum"].to_numpy(float), g["size"].to_numpy(float), blocks.to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            res["precision"] = interval(tp[idx].sum(1) / nd[idx].sum(1))
        res["fa_per_min"] = interval((nd - tp)[idx].sum(1) * 60 / nf[idx].sum(1))
        out[band] = res
    return out


def fmt(value: Any, ci: list[Any] | None = None, digits: int = 2) -> str:
    """``0.41 [0.28-0.55]`` (or ``-`` for a missing value)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    text = f"{value:.{digits}f}"
    if ci and ci[0] is not None:
        text += f" [{ci[0]:.{digits}f}-{ci[1]:.{digits}f}]"
    return text


def task_table(metrics: dict[str, Any], run: str) -> str:
    """The task's table (metric rows x band columns) of one run, as Markdown (intervals stay in metrics.json).

    Args:
        metrics: ``score`` output.
        run: Run name.

    Returns:
        Markdown text.
    """
    bands = ["0-200", "200-400"]
    head = (f"**{run}** @ conf >= {metrics['threshold']:.2f}; {metrics['frames']} frames "
            f"({metrics['minutes']:.1f} min of video); vehicles (by the band where they first appear) / label boxes: "
            + ", ".join(f"{b} m {metrics[b].get('vehicles', 0)} / {metrics[b]['label_boxes']}" for b in bands))
    lines = [head, "", "| Metric | 0-200 m | 200-400 m |", "|---|---|---|"]
    spec = [("Detection rate TP / (TP + FN)", "detection_rate", 2), ("Precision TP / (TP + FP)", "precision", 2),
            ("False alarms / min FP x 60 / N_frames", "fa_per_min", 2)]
    for name, key, d in spec:
        lines.append(f"| {name} | " + " | ".join(fmt(metrics[b][key], digits=d) for b in bands) + " |")
    ttfd = []
    for b in bands:
        t = metrics[b].get("ttfd_s") or {}
        never = t.get("never_detected")
        ttfd.append(fmt(t.get("mean")) + f"; p90 {fmt(t.get('p90'))}"
                    + (f"; never detected {never:.0%}" if never is not None else ""))
    lines.append("| Time to first detection, s (mean; p90; never detected) | " + " | ".join(ttfd) + " |")
    lines.append("| mAP@0.5 (bonus) | " + " | ".join(fmt(metrics[b]["map50"]) for b in bands) + " |")
    lines += ["", "Detection rate and precision count boxes over every frame (a vehicle seen in 100 frames and found in "
              "74 adds 74 TP and 26 FN). False alarms / min is the task's formula FP x 60 / N_frames, "
              "i.e. false alarms per 60 frames; per minute of video it is "
              + " / ".join(fmt(metrics[b]["fa_per_min_real"], digits=0) for b in bands) + ". "
              "TTFD counts from the first frame a vehicle is in the ground truth."
              + (" Matching: IoU >= 0.3 up to 200 m, NWD beyond; with IoU only the detection rate would be "
                 + " / ".join(fmt((metrics.get("iou_only_detection_rate") or {}).get(b)) for b in bands) + "."
                 if metrics.get("iou_only_detection_rate") else "")]
    lines += ["", "Per clip:", "", md_table(per_clip_table(metrics))]
    return "\n".join(lines) + "\n"


def md_table(df: pd.DataFrame) -> str:
    """A DataFrame as a Markdown table (no extra dependency)."""
    cells = [[("-" if pd.isna(v) else f"{v:.2f}" if isinstance(v, float) else str(v)) for v in row]
             for row in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(r) + " |" for r in cells])


def per_clip_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """Per-clip counts of one run: where the misses and false alarms are.

    Args:
        metrics: ``score`` output.

    Returns:
        One row per clip.
    """
    rows = [{"clip": c, **{k: v for k, v in m.items() if k != "fa_per_min_real"}} for c, m in metrics["per_clip"].items()]
    return pd.DataFrame(rows).rename(columns={"fa_per_min": "FA/min (FP x 60 / N_frames)"})


def stages_table(all_metrics: list[dict[str, Any]]) -> str:
    """Every scored run side by side (pipeline stages, then trained runs), as Markdown.

    Args:
        all_metrics: ``score`` outputs, in table order.

    Returns:
        Markdown text.
    """
    bands = ["0-200", "200-400"]
    cols = [("DR", "detection_rate"), ("P", "precision"), ("FA/min", "fa_per_min")]
    lines = ["| Stage | " + " | ".join(f"{c} {b}" for c, _ in cols for b in bands) + " | mAP@0.5 all | frames |",
             "|---|" + "---|" * (len(cols) * len(bands) + 2)]
    for m in all_metrics:
        base = m["run"].removesuffix(STAGE_FRAMES)
        name = STAGES.get(m["run"], f"trained {base} @ {m['threshold']:.2f}"
                          + (", stage frames only" if m["run"].endswith(STAGE_FRAMES) else ""))
        cells = [fmt(m[b][k]) for _, k in cols for b in bands]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {fmt(m['all']['map50'])} | {m['frames']} |")
    lines += ["", "FA/min = FP x 60 / N_frames (task formula). Stages 02-03b ran on "
              "every step-th frame (~10 fps) and are scored on those frames; the auto labels and the trained model on "
              "every frame (the 'stage frames only' row puts the model on the same frames as the stages). "
              "mAP of the fused rows ranks boxes by the mean confidence of the models that voted."]
    return "\n".join(lines) + "\n"


def add_bands(df: pd.DataFrame, frames: pd.DataFrame, ref_len: float, edges: list[float]) -> pd.DataFrame:
    """Distance (pinhole, from the box long side) and band of each box, with the frame time.

    Args:
        df: Boxes with ``clip_id, frame_idx, x1..y2``.
        frames: Frames with ``f_px`` and ``t_s``.
        ref_len: Reference vehicle length, m.
        edges: Band edges, m.

    Returns:
        ``df`` (only frames in ``frames``) with ``dist_m``, ``band``, ``t_s``.
    """
    df = df.drop(columns=[c for c in ("dist_m", "band", "t_s") if c in df.columns])
    df = df.merge(frames[["clip_id", "frame_idx", "f_px", "t_s"]], on=["clip_id", "frame_idx"])
    long_side = np.maximum(df["x2"] - df["x1"], df["y2"] - df["y1"])
    df["dist_m"] = df["f_px"] * ref_len / np.maximum(long_side, 1e-6)
    df["band"] = band_labels(df["dist_m"].to_numpy(), edges) if len(df) else []
    return df.drop(columns="f_px").reset_index(drop=True)


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    parser = base_arg_parser(__doc__)
    parser.add_argument("--runs", nargs="+", required=True,
                        help=f"Run names (outputs/08_train/<run>/weights/best.pt); '{BEST}' = the run chosen on val; "
                             f"'{STAGES_ALIAS}' = every auto-labelling stage, or one of {list(STAGES)} (eval).")
    parser.add_argument("--split", choices=["val", "eval"], default="val")
    parser.add_argument("--overwrite", action="store_true", help="Re-run inference even if the cached detections match.")
    parser.add_argument("--no-video", action="store_true", help="Skip the overlay videos and stills.")
    parser.add_argument("--video-clips", nargs="+", default=None,
                        help="Eval: clips to render as overlay videos instead of evaluate.viz.video_clips; 'all' = every "
                             "eval clip.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    stage = start_stage(f"09_eval/{args.split}", cfg, args)
    ecfg = cfg["evaluate"]
    runs = list(dict.fromkeys(args.runs))
    if BEST in runs:
        sel_path = stage.out_dir.parent / "val" / "selected.json"
        if args.split == "val" or not sel_path.is_file():
            raise ValueError(f"'{BEST}' needs --split eval and {sel_path} (run the val split first)")
        chosen = json.loads(sel_path.read_text())["run"]
        LOGGER.info("%s = %s (from %s)", BEST, chosen, sel_path)
        runs = list(dict.fromkeys(chosen if r == BEST else r for r in runs))
    if STAGES_ALIAS in runs:  # stages first, in pipeline order, then the trained runs
        runs = list(dict.fromkeys([*STAGES, *(r for r in runs if r != STAGES_ALIAS)]))
    camera, clips = load_clips(Path(cfg["paths"]["clips"]))
    edges = [float(e) for e in cfg["fuse"]["bands_m"]]
    datasets_root = Path(cfg["paths"]["datasets"])
    catalog = pd.read_parquet(cfg["paths"]["catalog"])
    frames = eval_frames(catalog, clips, args.split)

    tiers = pd.read_parquet(cfg["dataset"]["tiers_path"])
    if args.split == "val":
        boxes = pd.read_parquet(cfg["dataset"]["boxes_path"])
        val_ids = set(frames["clip_id"])
        labels = label_boxes(boxes[boxes["clip_id"].isin(val_ids)], tiers, cfg["dataset"], cfg["import"])
        steps = frame_steps(clips, int(cfg["predict"]["train_frame_step"]))
        if len(labels):  # auto labels exist every step-th frame -> every frame
            labels = pd.concat([interpolate_frames(g, int(steps[c])) for c, g in labels.groupby("clip_id")],
                               ignore_index=True)
    else:
        labels = pd.read_parquet(cfg["import"]["gt_path"])
    labels = add_bands(labels, frames, float(camera["ref_length_m"]), edges)
    LOGGER.info("%s: %d frames of %d clips (every frame), %d label boxes of %d vehicles", args.split, len(frames),
                frames["clip_id"].nunique(), len(labels),
                labels.groupby(["clip_id", "vehicle_id"]).ngroups)
    sel_bands = macro_bands(labels, ecfg)
    LOGGER.info("label boxes per band: %s; F1_macro over %s", labels["band"].value_counts().to_dict(), sel_bands)
    if len(sel_bands) < len(ecfg["select_bands"]):
        LOGGER.warning("bands with < %d label boxes are left out of F1_macro: %s", int(ecfg["select_min_boxes"]),
                       [b for b in ecfg["select_bands"] if b not in sel_bands])

    rows: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    ref_len = float(camera["ref_length_m"])
    train_step = int(cfg["predict"]["train_frame_step"])
    nwd_rule = ({"thr": float(ecfg["match_nwd"]), "c": float(cfg["fuse"]["nwd_c"]),
                 "near_max_m": float(cfg["fuse"]["near_max_m"])} if ecfg.get("match_nwd") is not None else None)

    def scored(name: str, preds: pd.DataFrame, fr_run: pd.DataFrame, lab_run: pd.DataFrame, thr: float | None
               ) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        """Matches, picks the threshold on val (``thr`` None), scores, logs and writes one run's results.

        Args:
            name: Result name (folder, table file, ``metrics.csv`` row).
            preds: Detections (``clip_id, frame_idx, x1..y2, conf``); only frames in ``fr_run`` count.
            fr_run: Frames to score.
            lab_run: Label boxes of those frames.
            thr: Confidence threshold; None = the best ``F1_macro`` on these labels.

        Returns:
            Matched detections, labels with ``c_op`` and the threshold used.
        """
        preds = add_bands(preds, fr_run, ref_len, edges)
        preds["m_op"], c_op = match(preds, lab_run, float(ecfg["match_iou"]), nwd_rule)
        preds["m_ap"], _ = match(preds, lab_run, 0.5)
        lab = lab_run.assign(c_op=c_op)
        curve_df = pd.DataFrame()
        if thr is None:
            curve_df = threshold_curve(preds, lab, sel_bands)
            thr = float(curve_df.loc[curve_df["f1_macro"].idxmax(), "thr"])
        metrics = score(preds, lab, fr_run, thr, ecfg)
        metrics["run"] = name
        if nwd_rule:  # the same detections matched by IoU only: what NWD changes
            _, c_iou = match(preds, lab_run, float(ecfg["match_iou"]))
            metrics["iou_only_detection_rate"] = {
                b: round(float((c_iou[m] >= thr).mean()), 4) for b in ["all", "0-200", "200-400"]
                if (m := (np.ones(len(lab), bool) if b == "all" else (lab["band"] == b).to_numpy())).any()}
        for band, c in bootstrap_ci(preds, lab, fr_run, thr, ["all", "0-200", "200-400"], int(cfg["seed"])).items():
            metrics[band]["ci95"] = c
        if len(curve_df):
            metrics["pr_curve"] = curve_df.round(4).to_dict(orient="list")
        row = {"run": name, "threshold": thr, "F1_macro": metrics["f1_macro"]}
        for band in ["all", "0-200", "200-400"]:
            m, t = metrics[band], metrics[band].get("ttfd_s") or {}
            row.update({f"DR_{band}": m["detection_rate"], f"P_{band}": m["precision"], f"F1_{band}": m["f1"],
                        f"FA/min_{band}": m["fa_per_min"], f"FA/min_real_{band}": m["fa_per_min_real"],
                        f"TTFD_mean_{band}": t.get("mean"), f"TTFD_p90_{band}": t.get("p90"),
                        f"never_{band}": t.get("never_detected"), f"mAP50_{band}": m["map50"]})
            for key, short in [("detection_rate", "DR"), ("precision", "P"), ("fa_per_min", "FA/min")]:
                row[f"{short}_lo_{band}"], row[f"{short}_hi_{band}"] = m["ci95"][key]
        LOGGER.info("%s @%.2f: %s", name, thr, {k: v for k, v in row.items() if k not in ("run", "threshold")})
        errors = error_breakdown(preds, lab, thr)
        metrics["errors"] = errors.to_dict(orient="records")
        out_dir = stage.out_dir / name
        out_dir.mkdir(exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        LOGGER.info("%s per clip @%.2f:\n%s", name, thr, per_clip_table(metrics).to_string(index=False))
        LOGGER.info("%s errors @%.2f (boxes per band):\n%s", name, thr,
                    errors.pivot_table(index=["what", "pass", "outcome"], columns="band", values="n", fill_value=0,
                                       aggfunc="sum").to_string())
        (stage.out_dir / f"table_{name}.md").write_text(task_table(metrics, name), encoding="utf-8")
        rows.append(row)
        all_metrics.append(metrics)
        return preds, lab, thr

    with_stages = args.split == "eval" and any(r in STAGES for r in runs)
    for run in runs:
        if run in STAGES:
            if args.split == "val":
                LOGGER.warning("%s on val is meaningless: the val labels are the auto labels; skipped", run)
                continue
            # boxes already exist (stages 02-05); the operating point is the stage's own rule
            preds, stage_thr = stage_predictions(run, cfg, clips, set(frames["clip_id"]))
            fr_run, lab_run = (frames, labels) if run == AUTOLABEL else processed_frames(frames, labels, clips,
                                                                                          train_step)
            LOGGER.info("%s: %d boxes on %d frames", run, len(preds), len(fr_run))
            scored(run, preds, fr_run, lab_run, stage_thr)
            continue

        run_dir = stage.out_dir / run
        run_dir.mkdir(exist_ok=True)
        weights = Path(cfg["paths"]["outputs"]) / "08_train" / run / "weights" / "best.pt"
        raw = cached_predictions(run_dir, weights, frames, cfg, datasets_root, args.overwrite)
        thr = None if args.split == "val" else val_threshold(stage.out_dir.parent / "val" / run / "metrics.json")
        preds, lab, thr = scored(run, raw, frames, labels, thr)
        if with_stages:  # the same model on the frames the stages saw, for a like-for-like stage table
            fr_s, lab_s = processed_frames(frames, labels, clips, train_step)
            scored(run + STAGE_FRAMES, raw, fr_s, lab_s, thr)
        render_error_sheets(run_dir / "viz", run, preds, lab, frames, thr, tiers, datasets_root)
        if not args.no_video:
            vcfg = ecfg.get("viz", {})
            if args.split == "val":  # diagnosis: the val clips themselves
                ids = sorted(frames["clip_id"].unique())
                vcfg = {**vcfg, "video_clips": ids, "example_clips": ids}
            elif args.video_clips:
                vcfg = {**vcfg, "video_clips": sorted(frames["clip_id"].unique()) if args.video_clips == ["all"]
                        else args.video_clips}
            overlay = overlay_boxes(preds, lab, thr)
            render_eval_viz(run_dir / "viz", overlay, frames, run, thr, vcfg, datasets_root)
            if args.split == "eval":  # auto labels | my review | the model, side by side
                auto = autolabel_predictions(cfg, clips, set(frames["clip_id"]))
                render_story(run_dir / "viz", frames, auto, review_boxes(auto, labels), overlay, run, thr, vcfg,
                             datasets_root)

    table = pd.DataFrame(rows)
    out_csv = stage.out_dir / "metrics.csv"
    if out_csv.is_file():
        old = pd.read_csv(out_csv)
        table = pd.concat([old[~old["run"].isin(table["run"])], table], ignore_index=True)
    table.to_csv(out_csv, index=False)
    if with_stages:
        (stage.out_dir / "table_stages.md").write_text(stages_table(all_metrics), encoding="utf-8")
        LOGGER.info("stages table -> %s\n%s", stage.out_dir / "table_stages.md", stages_table(all_metrics))
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 60):
        LOGGER.info("%s metrics:\n%s", args.split, table.set_index("run").T.to_string())
    current = table[table["run"].isin([r for r in runs if r not in STAGES])]  # never a stale row of another run
    if args.split == "val" and len(current):
        best = select_run(current, float(ecfg["select_tie"]))
        sel = {"run": best["run"], "threshold": float(best["threshold"]), "F1_macro": float(best["F1_macro"]),
               "F1_200-400": None if pd.isna(best["F1_200-400"]) else float(best["F1_200-400"]),
               "bands": sel_bands, "candidates": current[["run", "threshold", "F1_macro", "F1_0-200", "F1_200-400"]]
               .to_dict(orient="records"),
               "rule": f"max F1_macro over {sel_bands}; within {ecfg['select_tie']} the best F1_200-400"}
        (stage.out_dir / "selected.json").write_text(json.dumps(sel, indent=2, default=str), encoding="utf-8")
        LOGGER.info("selected: %s @%.2f (F1_macro %.4f) -> %s", sel["run"], sel["threshold"], sel["F1_macro"],
                    stage.out_dir / "selected.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
