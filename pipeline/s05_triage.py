"""Stage 05: decide per track (vehicle / suspect / two_wheeler) and export CVAT tasks for review.

Review happens per track, not per frame: deleting a wrong track in CVAT cleans all its
frames at once, so every processed frame of an exported clip goes to CVAT. Only the eval
clips are exported (``export_roles``): they become the ground truth; train / val keep the
auto labels (tiers A-C).

Track tier (thresholds picked in stage 04c from crop sheets, ~10 crops per score bin):
    two_wheeler  re-check (04b) calls it bicycle / motorcycle in >= half of its checks
                 (40 / 40 such crops were two-wheelers); not a vehicle in this task
    A_sure       score >= 0.34 (crops: 10 / 10 vehicles), any size (big trucks are real)
    B_likely     score >= 0.18 (~ two models agree) and a plausible size
    C_rescued    lower score, plausible size, and the re-check says vehicle in >= 2/3 of its
                 checks with mean confidence >= 0.3 (often small or vertical cars that the
                 COCO detectors see poorly)
    suspect      the rest: roofs, markings, panels, one-model flicker
Plausible size: 0.68-2.5x the vehicles moving nearby. Bigger: houses, roofs; smaller: panels,
markings, two-wheelers (in 300 blind-labelled hard tracks, 66 of 72 below 0.68x were not vehicles;
raising the bound from 0.25 kept the same cars and cut the junk ~2x).
The re-check only adds evidence (rescue, two-wheeler); no vote is never a reason to drop.
CVAT labels: vehicle (tiers A-C, ``tier`` attribute to filter by), suspect, two_wheeler.
Stage 06 keeps ``vehicle`` tracks only.

Tiers are decided for every clip.

Outputs (outputs/05_triage/):
    tracks.parquet                 one row per track: features, tier, label
    cvat/labels.json               labels + attributes, paste into CVAT task constructor (Raw)
    cvat/<clip>.zip                all processed frames of the clip: CVAT task data
    cvat/<clip>/annotations.xml    pre-labels, "Upload annotations" -> CVAT for video 1.1
    cvat/<clip>/frames.csv         CVAT frame number -> clip frame, time, boxes per label
    stats.json                     tracks per tier and label, vehicle tracks per band
    viz/<clip>.mp4                 whole clip: kept tracks (A green, B yellow, C cyan) over removed ones
                                   (thin red suspect, magenta two-wheeler), legend with track counts

Example:
    python -m pipeline.s05_triage
"""

from __future__ import annotations

import json
import logging
import zipfile
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.common import base_arg_parser, frame_steps, load_clips, load_yaml, start_stage
from pipeline.s02_predict import select_frames
from pipeline.s04c_analysis import track_features
from pipeline.viz import render_boxes_video

LOGGER = logging.getLogger("pipeline.s05_triage")

LABELS = ("vehicle", "suspect", "two_wheeler")
# video style per tier: BGR colour, label, line width (removed tracks thin, drawn first)
TIER_STYLE: dict[str, tuple[tuple[int, int, int], str, int]] = {
    "suspect": ((60, 60, 230), "", 1), "two_wheeler": ((220, 60, 220), "2w", 1),
    "C_rescued": ((255, 200, 0), "C", 2), "B_likely": ((0, 220, 255), "B", 2), "A_sure": ((60, 200, 60), "A", 2)}
TIER_LABEL = {"A_sure": "vehicle", "B_likely": "vehicle", "C_rescued": "vehicle", "suspect": "suspect",
              "two_wheeler": "two_wheeler"}
ATTRIBUTES: list[dict[str, Any]] = [
    {"name": "auto", "mutable": True, "input_type": "checkbox", "default_value": "false", "values": ["false"]},
    {"name": "tier", "mutable": True, "input_type": "text", "default_value": "", "values": [""]},
    {"name": "k", "mutable": True, "input_type": "number", "default_value": "0", "values": ["0", "3", "1"]},
    {"name": "k_zoom", "mutable": True, "input_type": "number", "default_value": "0", "values": ["0", "3", "1"]},
    {"name": "models", "mutable": True, "input_type": "text", "default_value": "", "values": [""]},
    {"name": "flag", "mutable": True, "input_type": "text", "default_value": "", "values": [""]},
    {"name": "decoy", "mutable": True, "input_type": "text", "default_value": "", "values": [""]},
]


def decide_tracks(tracks: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Tier and label of every track (see module docstring).

    Args:
        tracks: Track features (stage 04c ``track_features``) with re-check columns (NaN if missing).
        cfg: ``triage`` config section.

    Returns:
        ``tracks`` with ``tier`` and ``label``.
    """
    lo, hi = (float(v) for v in cfg["size_range"])
    size_ok = tracks["size_ratio"].isna() | tracks["size_ratio"].between(lo, hi)  # no reference -> no size verdict
    two_wheeler = tracks["rc_two_wheeler"].fillna(0) >= float(cfg["two_wheeler_share"])
    rescued = ((tracks["rc_vehicle"].fillna(0) >= float(cfg["rescue_share"]))
               & (tracks["rc_vehicle_conf"].fillna(0) >= float(cfg["rescue_conf"])))
    score = tracks["score"]
    tier = np.select(
        [two_wheeler, score >= float(cfg["score_sure"]), (score >= float(cfg["score_likely"])) & size_ok,
         size_ok & rescued],
        ["two_wheeler", "A_sure", "B_likely", "C_rescued"], "suspect")
    return tracks.assign(tier=tier, label=pd.Series(tier, index=tracks.index).map(TIER_LABEL))


def write_cvat_task(out_dir: Path, clip_id: str, frames: pd.DataFrame, boxes: pd.DataFrame, k_col: str,
                    datasets_root: Path) -> pd.DataFrame:
    """Writes the image archive, CVAT-for-video XML with tracks, and the frame map of one clip.

    CVAT numbers task frames by sorted file name; frame files are ``{clip}_{idx:06d}.jpg``,
    so CVAT frame i is row i of ``frames`` (sorted by frame_idx).

    Args:
        out_dir: ``cvat/<clip>`` folder.
        clip_id: Clip ID (task name).
        frames: Processed frames of the clip, sorted by ``frame_idx``.
        boxes: Tracked boxes of the clip with a ``label`` column.
        k_col: Vote column.
        datasets_root: Root for catalog paths.

    Returns:
        Frame map (``cvat_frame, frame_idx, t_s`` and boxes per label).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_dir.with_suffix(".zip"), "w", zipfile.ZIP_STORED) as zf:  # JPEGs: no recompression
        for path in frames["path"]:
            src = datasets_root / path
            zf.write(src, arcname=src.name)

    cvat_frame = {f: i for i, f in enumerate(frames["frame_idx"])}
    width, height = int(frames["width"].iloc[0]), int(frames["height"].iloc[0])
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    task = ET.SubElement(ET.SubElement(root, "meta"), "task")
    ET.SubElement(task, "name").text = clip_id
    ET.SubElement(task, "size").text = str(len(frames))
    ET.SubElement(task, "mode").text = "interpolation"
    labels_node = ET.SubElement(task, "labels")
    for name in LABELS:
        label = ET.SubElement(labels_node, "label")
        ET.SubElement(label, "name").text = name
        attrs = ET.SubElement(label, "attributes")
        for a in ATTRIBUTES:
            node = ET.SubElement(attrs, "attribute")
            for key in ("name", "mutable", "input_type", "default_value"):
                ET.SubElement(node, key).text = str(a[key])
            ET.SubElement(node, "values").text = "\n".join(a["values"])

    sel = boxes[boxes["frame_idx"].isin(cvat_frame)].copy()
    sel["cvat_frame"] = sel["frame_idx"].map(cvat_frame)
    sel["tid"] = np.where(sel["track_id"] >= 0, sel["track_id"], -1 - np.arange(len(sel)))  # untracked -> own track
    for xml_id, (_, trk) in enumerate(sel.sort_values("cvat_frame").groupby("tid", sort=True)):
        node = ET.SubElement(root, "track", id=str(xml_id), label=str(trk["label"].iloc[0]), source="auto")
        present = trk.set_index("cvat_frame")
        frames_span = range(int(present.index.min()), min(int(present.index.max()) + 2, len(frames)))
        last = None
        for f in frames_span:
            if f in present.index:
                r = present.loc[f]
                last = r
                box = ET.SubElement(node, "box", frame=str(f), outside="0", occluded="0", keyframe="1",
                                    xtl=f"{max(r.x1, 0):.2f}", ytl=f"{max(r.y1, 0):.2f}",
                                    xbr=f"{min(r.x2, width):.2f}", ybr=f"{min(r.y2, height):.2f}", z_order="0")
                values = {"auto": "true", "tier": r["tier"], "k": int(r["k"]), "k_zoom": int(r[k_col]),
                          "models": r["models"],
                          "flag": r.get("track_flag", ""), "decoy": r.get("zoom_decoy", "") or ""}
                for name, value in values.items():
                    ET.SubElement(box, "attribute", name=name).text = str(value)
            elif last is not None:  # object absent here: close the track until the next keyframe
                ET.SubElement(node, "box", frame=str(f), outside="1", occluded="0", keyframe="1",
                              xtl=f"{max(last.x1, 0):.2f}", ytl=f"{max(last.y1, 0):.2f}",
                              xbr=f"{min(last.x2, width):.2f}", ybr=f"{min(last.y2, height):.2f}", z_order="0")
                last = None
    ET.indent(root)
    ET.ElementTree(root).write(out_dir / "annotations.xml", encoding="utf-8", xml_declaration=True)

    counts = sel.groupby(["frame_idx", "label"]).size().unstack(fill_value=0).reindex(columns=list(LABELS), fill_value=0)
    fmap = frames[["clip_id", "frame_idx", "t_s"]].assign(cvat_frame=range(len(frames)))
    fmap = fmap.join(counts.add_prefix("n_"), on="frame_idx").fillna({f"n_{n}": 0 for n in LABELS})
    fmap = fmap.astype({f"n_{n}": int for n in LABELS})
    fmap[["cvat_frame", "clip_id", "frame_idx", "t_s", *[f"n_{n}" for n in LABELS]]].to_csv(out_dir / "frames.csv",
                                                                                         index=False)
    return fmap


def render_tier_videos(out_dir: Path, frames_all: pd.DataFrame, boxes: pd.DataFrame, tracks: pd.DataFrame,
                       datasets_root: Path) -> None:
    """Whole-clip videos of what triage keeps and removes: one per clip, kept tracks on top.

    Args:
        out_dir: Folder for ``<clip>.mp4``.
        frames_all: Processed frames of all clips.
        boxes: Tracked boxes with ``tier``.
        tracks: Track table with ``tier``.
        datasets_root: Root for catalog paths.
    """
    order = {t: i for i, t in enumerate(TIER_STYLE)}
    for clip_id, frames in frames_all.groupby("clip_id", sort=True):
        cb = boxes[boxes["clip_id"] == clip_id]
        cb = cb.iloc[np.argsort(cb["tier"].map(order).fillna(0).to_numpy(), kind="stable")]
        n = tracks.loc[tracks["clip_id"] == clip_id, "tier"].value_counts()
        legend = [(f"{name} {n.get(tier, 0)} tracks", TIER_STYLE[tier][0]) for tier, name in
                  (("A_sure", "A sure"), ("B_likely", "B likely"), ("C_rescued", "C rescued"),
                   ("two_wheeler", "two-wheeler (removed)"), ("suspect", "suspect (removed)"))]

        def style(r: Any) -> tuple[tuple[int, int, int], str, int]:
            color, letter, thick = TIER_STYLE.get(r.tier, ((255, 255, 255), "", 1))
            return color, (f"{letter}{r.track_id}" if letter else ""), thick

        render_boxes_video(out_dir / f"{clip_id}.mp4", frames.sort_values("frame_idx"), cb, datasets_root, 0, style,
                           legend)


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    parser = base_arg_parser(__doc__)
    parser.add_argument("--input-path", type=Path, default=None, help="Stage 04 tracked.parquet (default: from config).")
    parser.add_argument("--no-video", action="store_true", help="Skip the per-clip review videos.")
    parser.add_argument("--export-roles", nargs="+", default=None,
                        help="Roles to export as CVAT tasks, e.g. train val eval (default: triage.export_roles).")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    stage = start_stage("05_triage", cfg, args)
    tcfg = cfg["triage"]
    _, clips = load_clips(Path(cfg["paths"]["clips"]))
    roles = {c.clip_id: c.role for c in clips}
    datasets_root = Path(cfg["paths"]["datasets"])
    boxes = pd.read_parquet(args.input_path or Path(tcfg["input_path"]))
    k_col = "k_zoom" if "k_zoom" in boxes.columns else "k"
    catalog = pd.read_parquet(cfg["paths"]["catalog"])
    frames_all = select_frames(catalog, frame_steps(clips, int(cfg["predict"]["train_frame_step"])),
                               [c.clip_id for c in clips])  # also clips without boxes (empty scenes)

    tracks, _, _ = track_features(boxes, frames_all, cfg["analysis"], [float(e) for e in cfg["fuse"]["bands_m"]])
    rc_path = Path(tcfg["recheck_path"])
    rc_cols = ["rc_vehicle", "rc_two_wheeler", "rc_vehicle_conf"]
    if rc_path.is_file():
        tracks = tracks.merge(pd.read_parquet(rc_path)[["clip_id", "track_id", *rc_cols]], on=["clip_id", "track_id"],
                              how="left")
    else:
        LOGGER.warning("No re-check results at %s: no two-wheeler filter, no rescue", rc_path)
        tracks[rc_cols] = np.nan
    tracks = decide_tracks(tracks, tcfg)
    tracks.insert(1, "role", tracks["clip_id"].map(roles))
    tracks.to_parquet(stage.out_dir / "tracks.parquet", index=False)
    boxes = boxes.merge(tracks[["clip_id", "track_id", "label", "tier"]], on=["clip_id", "track_id"], how="left")
    boxes["label"] = boxes["label"].fillna("vehicle")  # untracked boxes (none expected) stay visible
    boxes["tier"] = boxes["tier"].fillna("")

    cvat_dir = stage.out_dir / "cvat"
    cvat_dir.mkdir(exist_ok=True)
    (cvat_dir / "labels.json").write_text(
        json.dumps([{"name": n, "type": "rectangle", "attributes": ATTRIBUTES} for n in LABELS], indent=2),
        encoding="utf-8")  # CVAT >= 2.x rejects labels without a type
    export_roles = set(args.export_roles or tcfg["export_roles"])
    stats: dict[str, Any] = {}
    for clip_id, frames in frames_all.groupby("clip_id", sort=True):
        frames = frames.sort_values("frame_idx").reset_index(drop=True)
        role = roles[clip_id]
        clip_boxes = boxes[boxes["clip_id"] == clip_id]
        clip_tracks = tracks[tracks["clip_id"] == clip_id]
        exported = role in export_roles
        if exported:
            write_cvat_task(cvat_dir / clip_id, clip_id, frames, clip_boxes, k_col, datasets_root)
        vehicle = clip_tracks[clip_tracks["label"] == "vehicle"]
        stats[clip_id] = {
            "role": role,
            "exported": exported,
            "frames": len(frames),
            "tracks": clip_tracks["tier"].value_counts().to_dict(),
            "boxes": clip_boxes["label"].value_counts().to_dict(),
            "vehicle_tracks_by_band": vehicle["band"].value_counts().sort_index().to_dict(),
        }
        LOGGER.info("%s (%s): %d frames, tracks %s, vehicle by band %s%s", clip_id, role, len(frames),
                    stats[clip_id]["tracks"], stats[clip_id]["vehicle_tracks_by_band"], "" if exported else " (no export)")

    by_role = tracks.groupby(["role", "tier"]).size().unstack(fill_value=0)
    stats["_tracks_by_role"] = {r: row.to_dict() for r, row in by_role.iterrows()}
    stage.save_stats(stats)
    LOGGER.info("Tracks by role:\n%s", by_role.to_string())
    LOGGER.info("CVAT tasks -> %s", cvat_dir)
    if tcfg.get("video", True) and not args.no_video:
        render_tier_videos(stage.viz_dir, frames_all, boxes, tracks, datasets_root)
        LOGGER.info("Videos (green A, yellow B, cyan C kept; thin red removed, magenta two-wheeler) -> %s",
                    stage.viz_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
