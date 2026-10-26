"""Stage 04: link fused boxes into tracks, fill short gaps, tell moving from static.

Why a tracker: a real vehicle lives over many frames, a false positive flickers.
Why our own and not ByteTrack: our boxes are already fused and scored,
ByteTrack's second (low-score) association is hard-wired to IoU, and far vehicles need
NWD; we also need camera-motion compensation anyway for the static check.

Per clip, frame by frame (the frames stage 02 processed):
    1. camera motion: homography previous -> current frame (ORB + RANSAC on a downscaled
       frame; moving vehicles are outliers). Track boxes are warped with it; a failed estimate
       reuses the last good one (never "camera stands still"),
    2. predict each track: warped last box + its own velocity,
    3. associate predictions with boxes (Hungarian; IoU for near, NWD for far, looser
       thresholds than fusion), unmatched boxes start new tracks, a track ends after
       ``max_gap`` missed frames,
    4. per track: duration, own speed in car lengths / s (camera motion removed),
       gaps <= max_gap are filled with interpolated boxes (source = tracker).
Box flag: stable_moving | stable_static | short | flicker | interpolated.
Nothing is deleted here; flags feed triage.

Outputs (outputs/04_track/):
    tracked.parquet  boxes of stage 03/03b + track_id, track_flag, track_duration_s, track_speed, interpolated
    tracks.parquet   one row per track (speed median + p10 / p90)
    stats.json       per clip: camera-motion fallback share and shift, tracks per type,
                     duration / speed / gap distributions
    viz/<clip>.mp4   every clip (``viz_seconds``, 0 = whole clip), colour = flag, label = track id + speed

Example:
    python -m pipeline.s04_track
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from pipeline.common import base_arg_parser, frame_steps, load_clips, load_yaml, start_stage
from pipeline.s02_predict import iter_images, select_frames
from pipeline.s03_fuse import match_scores
from pipeline.viz import render_boxes_video

LOGGER = logging.getLogger("pipeline.s04_track")

FLAG_BGR = {"stable_moving": (60, 170, 60), "stable_static": (0, 160, 255), "short": (200, 120, 40),
            "flicker": (40, 40, 230), "interpolated": (200, 60, 200)}


MIN_POINTS = 20  # keypoints / matches / RANSAC inliers needed to trust a homography


def camera_homography(prev: np.ndarray, cur: np.ndarray, max_side: int) -> tuple[np.ndarray | None, str]:
    """Homography mapping previous-frame pixels to current-frame pixels.

    ORB features on downscaled grayscale frames, RANSAC rejects moving vehicles.
    A result with too few inliers or a frame-to-frame zoom outside 0.8-1.25x is rejected:
    the caller then reuses the last good homography (camera motion is smooth) instead of
    silently assuming a still camera, which would make followed vehicles look static.

    Args:
        prev: Previous BGR frame.
        cur: Current BGR frame.
        max_side: Long side the frames are downscaled to.

    Returns:
        ``(3, 3)`` homography in full-resolution pixels (``None`` on failure) and a status:
        ``ok``, ``few_features``, ``few_matches``, ``few_inliers`` or ``bad_scale``.
    """
    s = min(1.0, max_side / max(prev.shape[:2]))
    a = cv2.cvtColor(cv2.resize(prev, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(cur, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(2000)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < MIN_POINTS or len(kb) < MIN_POINTS:
        return None, "few_features"
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(matches) < MIN_POINTS:
        return None, "few_matches"
    src = np.float32([ka[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if h is None or inliers is None or int(inliers.sum()) < MIN_POINTS:
        return None, "few_inliers"
    if not 0.8 <= np.sqrt(abs(np.linalg.det(h[:2, :2]))) <= 1.25:
        return None, "bad_scale"
    scale = np.diag([s, s, 1.0])
    return np.linalg.inv(scale) @ h @ scale, "ok"


def clip_homographies(paths: list[Path], max_side: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Camera motion for every processed frame of a clip, with diagnostics.

    Args:
        paths: Frame images in order.
        max_side: Long side for feature matching.

    Returns:
        ``H[i]`` mapping frame ``i-1`` to ``i`` (``H[0]`` = identity; a failed estimate reuses
        the last good one) and stats: status counts, fallback share, camera shift quantiles
        (px per processed frame, displacement of the frame centre).
    """
    homographies, statuses, shifts = [np.eye(3)], [], []
    prev, last_good = None, np.eye(3)
    for img in iter_images(paths):
        if prev is not None:
            h, status = camera_homography(prev, img, max_side)
            statuses.append(status)
            if h is not None:
                last_good = h
                c = np.float32([[[img.shape[1] / 2, img.shape[0] / 2]]])
                shifts.append(float(np.linalg.norm(cv2.perspectiveTransform(c, h)[0, 0] - c[0, 0])))
            homographies.append(last_good)
        prev = img
    counts = pd.Series(statuses, dtype=str).value_counts().to_dict()
    return homographies, {
        "status_counts": counts,
        "fallback_share": round(1 - counts.get("ok", 0) / max(len(statuses), 1), 4),
        "shift_px_quantiles": quantiles(pd.Series(shifts, dtype=float)),
    }


def warp_box(box: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Applies a homography to a box (axis-aligned hull of the warped corners).

    Args:
        box: ``(4,)`` xyxy box.
        h: ``(3, 3)`` homography.

    Returns:
        Warped ``(4,)`` xyxy box.
    """
    corners = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]], np.float32)
    pts = cv2.perspectiveTransform(corners[None], h)[0]
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])


def center(box: np.ndarray) -> np.ndarray:
    """Centre ``(2,)`` of an xyxy box."""
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


@dataclass
class Track:
    """State of one track while a clip is being processed.

    Attributes:
        tid: Track id (unique within the clip).
        box: Last observed box, warped into the current frame (camera motion removed).
        v: Own velocity in px per processed frame (current-frame coordinates).
        misses: Frames since the last observation.
        rows: Indices of the observed boxes in the input table.
        speeds: Own speed per observation step, car lengths per second.
    """

    tid: int
    box: np.ndarray
    v: np.ndarray = field(default_factory=lambda: np.zeros(2))
    misses: int = 0
    rows: list[int] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)


def track_clip(boxes: pd.DataFrame, frames: pd.DataFrame, homographies: list[np.ndarray],
               cfg: dict[str, Any], fuse_cfg: dict[str, Any], ref_length_m: float) -> tuple[pd.Series, list[Track]]:
    """Runs the tracker over one clip.

    Args:
        boxes: Boxes of the clip (input rows, any order).
        frames: Processed frames of the clip, sorted (need ``frame_idx``, ``t_s``, ``f_px``).
        homographies: ``H[i]`` maps frame ``i-1`` to frame ``i`` (``H[0]`` = identity).
        cfg: ``track`` config section.
        fuse_cfg: ``fuse`` config section (near/far rule, NWD constant).
        ref_length_m: Reference vehicle length in metres.

    Returns:
        Track id per input row (index-aligned) and all tracks.
    """
    match_cfg = dict(fuse_cfg, iou_thr=cfg["iou_thr"], nwd_thr=cfg["nwd_thr"])
    by_frame = {k: g for k, g in boxes.groupby("frame_idx")}
    active: list[Track] = []
    done: list[Track] = []
    next_id = 0
    t_prev: dict[int, float] = {}  # track id -> time of last observation
    for i, fr in enumerate(frames.itertuples(index=False)):
        for tr in active:
            tr.box = warp_box(tr.box, homographies[i])
        g = by_frame.get(fr.frame_idx)
        det = g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64) if g is not None else np.zeros((0, 4))
        rows = g.index.to_numpy() if g is not None else np.zeros(0, int)

        matched_t, matched_d = [], []
        if active and len(det):
            cost = np.full((len(active), len(det)), 1e6)
            for a, tr in enumerate(active):
                pred = tr.box + np.tile(tr.v * (tr.misses + 1), 2)
                scores, passed = match_scores(pred, det, float(fr.f_px), match_cfg, ref_length_m)
                cost[a, passed] = 1.0 - scores[passed]
            for a, d in zip(*linear_sum_assignment(cost)):
                if cost[a, d] < 1e6:
                    matched_t.append(a)
                    matched_d.append(d)

        for a, d in zip(matched_t, matched_d):
            tr = active[a]
            steps = tr.misses + 1
            v_new = (center(det[d]) - center(tr.box)) / steps
            tr.v = v_new if len(tr.rows) == 1 else 0.5 * (tr.v + v_new)
            long_side = max(det[d][2] - det[d][0], det[d][3] - det[d][1], 1e-6)
            dt = max(fr.t_s - t_prev[tr.tid], 1e-6)
            tr.speeds.append(float(np.linalg.norm(v_new) * steps / dt / long_side))
            tr.box, tr.misses = det[d].copy(), 0
            tr.rows.append(int(rows[d]))
            t_prev[tr.tid] = fr.t_s

        still: list[Track] = []
        for a, tr in enumerate(active):
            if a not in matched_t:
                tr.misses += 1
                if tr.misses > int(cfg["max_gap"]):
                    done.append(tr)
                    continue
            still.append(tr)
        active = still
        for d in sorted(set(range(len(det))) - set(matched_d)):
            active.append(Track(tid=next_id, box=det[d].copy(), rows=[int(rows[d])]))
            t_prev[next_id] = fr.t_s
            next_id += 1

    tracks = done + active
    track_id = pd.Series(-1, index=boxes.index, dtype=int)
    for tr in tracks:
        track_id.loc[tr.rows] = tr.tid
    return track_id, tracks


def interpolate_gaps(rows: pd.DataFrame, frames: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """Linear boxes for frames a track skipped (gap <= max_gap processed frames).

    Args:
        rows: Observed boxes of one track, sorted by frame.
        frames: Processed frames of the clip, sorted.
        max_gap: Longest gap to fill.

    Returns:
        Interpolated rows (same columns, ``k = 0``, ``models = "tracker"``).
    """
    pos = {f: i for i, f in enumerate(frames["frame_idx"])}
    out = []
    for (_, a), (_, b) in zip(rows.iterrows(), rows.iloc[1:].iterrows()):
        ia, ib = pos[a["frame_idx"]], pos[b["frame_idx"]]
        if 1 < ib - ia <= max_gap + 1:
            for j in range(ia + 1, ib):
                w = (j - ia) / (ib - ia)
                new = a.copy()
                for c in ("x1", "y1", "x2", "y2"):
                    new[c] = (1 - w) * a[c] + w * b[c]
                new["frame_idx"] = frames["frame_idx"].iloc[j]
                out.append(new)
    return pd.DataFrame(out)


def classify(tr_rows: pd.DataFrame, duration_s: float, speed: float, cfg: dict[str, Any], k_col: str) -> str:
    """Flag of a track from its duration, votes and own speed.

    Args:
        tr_rows: Observed boxes of the track.
        duration_s: Time from first to last observation.
        speed: Median own speed, car lengths per second.
        cfg: ``track`` config section.
        k_col: Vote column (``k_zoom`` if zoom ran, else ``k``).

    Returns:
        One of ``stable_moving``, ``stable_static``, ``short``, ``flicker``.
    """
    if duration_s < float(cfg["min_track_s"]):
        return "flicker" if (tr_rows[k_col] <= 1).all() else "short"
    return "stable_moving" if speed >= float(cfg["static_speed"]) else "stable_static"


def flag_style(r: Any) -> tuple[tuple[int, int, int], str]:
    """Stage 04 video style: colour = track flag, label = track id + own speed (car lengths / s)."""
    label = str(r.track_id) if r.interpolated else f"{r.track_id} {r.track_speed:.1f}"
    return FLAG_BGR.get(r.track_flag, (255, 255, 255)), label


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    parser = base_arg_parser(__doc__)
    parser.add_argument("--input-path", type=Path, default=None, help="Stage 03/03b boxes (default: from config).")
    parser.add_argument("--clips", nargs="*", default=None, help="Only these clip IDs (default: all).")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    stage = start_stage("04_track", cfg, args)
    tcfg, fcfg = cfg["track"], cfg["fuse"]
    camera, clips = load_clips(Path(cfg["paths"]["clips"]))
    ref_len = float(camera["ref_length_m"])
    datasets_root = Path(cfg["paths"]["datasets"])
    boxes = pd.read_parquet(args.input_path or Path(tcfg["input_path"]))
    k_col = "k_zoom" if "k_zoom" in boxes.columns else "k"
    catalog = pd.read_parquet(cfg["paths"]["catalog"])
    # every registered clip, also one without boxes: an empty scene is what FA/min is measured on
    frames_all = select_frames(catalog, frame_steps(clips, int(cfg["predict"]["train_frame_step"])),
                               args.clips or [c.clip_id for c in clips])
    for clip_id in sorted(set(frames_all["clip_id"]) - set(boxes["clip_id"])):
        LOGGER.warning("%s: no boxes (empty scene, or stage 02 did not run on it)", clip_id)

    tracked_parts, track_rows, stats = [], [], {}
    for clip_id, frames in frames_all.groupby("clip_id", sort=True):
        frames = frames.sort_values("frame_idx").reset_index(drop=True)
        clip_boxes = boxes[boxes["clip_id"] == clip_id]
        homographies, camera_stats = clip_homographies([datasets_root / p for p in frames["path"]],
                                                       int(tcfg["homography_max_side"]))
        track_id, tracks = track_clip(clip_boxes, frames, homographies, tcfg, fcfg, ref_len)
        clip_boxes = clip_boxes.assign(track_id=track_id, interpolated=False, track_flag="",
                                       track_duration_s=np.nan, track_speed=np.nan)

        t_of = dict(zip(frames["frame_idx"], frames["t_s"]))
        pos = pd.Series(np.arange(len(frames)), index=frames["frame_idx"])
        extra, gaps = [], []
        for tr in tracks:
            rows = clip_boxes.loc[tr.rows].sort_values("frame_idx")
            duration = t_of[rows["frame_idx"].iloc[-1]] - t_of[rows["frame_idx"].iloc[0]]
            speed = float(np.median(tr.speeds)) if tr.speeds else 0.0
            flag = classify(rows, duration, speed, tcfg, k_col)
            clip_boxes.loc[tr.rows, ["track_flag", "track_duration_s", "track_speed"]] = [flag, duration, speed]
            gaps += list(np.diff(pos.loc[rows["frame_idx"]].to_numpy()) - 1)
            interp = interpolate_gaps(rows, frames, int(tcfg["max_gap"]))
            if len(interp):
                interp = interp.assign(k=0, models="tracker", interpolated=True, track_flag="interpolated")
                if k_col != "k":
                    interp[k_col] = 0
                conf_cols = [c for c in interp.columns if c.startswith(("conf_", "zoom_conf_"))]
                interp[conf_cols] = np.nan  # geometry is interpolated, no model saw this box
                extra.append(interp)
            sp = np.asarray(tr.speeds) if tr.speeds else np.full(1, np.nan)
            track_rows.append({"clip_id": clip_id, "track_id": tr.tid, "n_obs": len(rows), "duration_s": duration,
                               "speed_cl_s": speed, "speed_p10": float(np.percentile(sp, 10)),
                               "speed_p90": float(np.percentile(sp, 90)), "max_k": int(rows[k_col].max()),
                               "share_k1": float((rows[k_col] <= 1).mean()), "flag": flag})
        clip_tracked = pd.concat([clip_boxes, *extra], ignore_index=True)
        tracked_parts.append(clip_tracked)

        tr_df = pd.DataFrame([r for r in track_rows if r["clip_id"] == clip_id],
                             columns=["clip_id", "track_id", "duration_s", "speed_cl_s", "share_k1", "flag"])
        reliable = tr_df[tr_df["share_k1"] < 0.5]
        stats[clip_id] = {
            "frames": len(frames),
            "camera": camera_stats,
            "tracks": len(tr_df),
            "tracks_by_flag": tr_df["flag"].value_counts().to_dict(),
            "boxes_by_flag": clip_tracked["track_flag"].value_counts().to_dict(),
            "duration_s_quantiles_reliable_tracks": quantiles(reliable["duration_s"]),
            "duration_s_quantiles_k1_tracks": quantiles(tr_df[tr_df["share_k1"] >= 0.5]["duration_s"]),
            "speed_quantiles_reliable_tracks": quantiles(reliable["speed_cl_s"]),
            "gap_length_counts": pd.Series(gaps).value_counts().sort_index().to_dict() if gaps else {},
        }
        LOGGER.info("%s: %d frames, %d tracks %s, camera fallback %.1f%%, shift px %s", clip_id, len(frames),
                    len(tr_df), stats[clip_id]["tracks_by_flag"], 100 * camera_stats["fallback_share"],
                    camera_stats["shift_px_quantiles"])

    tracked = pd.concat(tracked_parts, ignore_index=True)
    tracked.to_parquet(stage.out_dir / "tracked.parquet", index=False)
    pd.DataFrame(track_rows).to_parquet(stage.out_dir / "tracks.parquet", index=False)
    stage.save_stats(stats)
    for clip_id, frames in frames_all.groupby("clip_id", sort=True):
        render_boxes_video(stage.viz_dir / f"{clip_id}.mp4", frames.sort_values("frame_idx"),
                           tracked[tracked["clip_id"] == clip_id], datasets_root, float(tcfg["viz_seconds"]), flag_style)
    LOGGER.info("Viz videos -> %s", stage.viz_dir)
    return 0


def quantiles(values: pd.Series) -> dict[str, float]:
    """10/50/90 % quantiles, rounded.

    Args:
        values: Numbers.

    Returns:
        ``{"p10": .., "p50": .., "p90": ..}`` (empty if no values).
    """
    if values.empty:
        return {}
    return {f"p{q}": round(float(values.quantile(q / 100)), 2) for q in (10, 50, 90)}


if __name__ == "__main__":
    sys.exit(main())
