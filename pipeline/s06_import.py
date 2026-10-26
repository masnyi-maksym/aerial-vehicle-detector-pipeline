"""Stage 06: reviewed CVAT exports -> ground truth of the eval clips (label ``vehicle`` only).

Input: the export zips as downloaded from CVAT (``CVAT for video 1.1`` for the tracks,
optionally ``CVAT for images 1.1`` for the frame file names), loose or packed in one outer zip.
Per task the latest export wins (task ``updated`` time, then ``dumped`` time); identical
re-downloads ("... (1).zip") collapse into one. The task name is the clip ID (stage 05).

CVAT frame i -> clip frame: file names of the images export (``<clip>_<frame_idx>.jpg``);
without one, the i-th processed frame of the clip (stage 05 sorts the same way).

Only boxes inside the image (``outside = 0``) of the kept labels are ground truth; interpolated
boxes CVAT wrote between keyframes count too (they are what the reviewer saw). Tracks keep their
CVAT id, so metrics can count distinct vehicles and time to first detection.

Review report (what the reviewer changed): per clip and pre-label tier, how many auto tracks
survived (``source=file`` with the stage 05 ``tier`` attribute), how many were added by hand
(``source=manual``), and ground-truth vehicles per distance band.

Outputs:
    datasets/annotations/eval_gt.parquet   clip_id, frame_idx, track_id, vehicle_id, x1..y2, keyframe, source,
                                           tier, dist_m, band, observed (one vehicle may span several tracks).
                                           Every frame of the clip: boxes between the labelled frames
                                           (every step-th) are interpolated per vehicle (observed=False)
    outputs/06_import/exports.csv          every export found and which one was used
    outputs/06_import/stats.json           per clip: tracks / boxes, review changes, vehicles per band
    outputs/06_import/dropped.parquet      double boxes removed (duplicate / inside / container) + the kept box
    outputs/06_import/viz/dropped.jpg      examples: green kept, red dropped (needs the frames on disk)
    outputs/06_import/viz/<clip>_gt.mp4    whole clip with the final ground truth (green auto, cyan hand-drawn,
                                           thin red removed doubles), label = vehicle id

Example:
    python -m pipeline.s06_import --exports ~/labels.zip
"""

from __future__ import annotations

import io
import logging
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import pandas as pd

from pipeline.common import base_arg_parser, frame_steps, load_clips, load_yaml, start_stage, vehicle_bands
from pipeline.geometry import contained_fraction, distance_m, focal_px, iou_one_to_many
from pipeline.s02_predict import select_frames
from pipeline.s03_fuse import band_labels
from pipeline.viz import render_boxes_video

LOGGER = logging.getLogger("pipeline.s06_import")
FRAME_NAME = re.compile(r"^(?P<clip>[^_/]+)_(?P<idx>\d+)\.\w+$")


@dataclass
class Export:
    """One CVAT export file.

    Attributes:
        source: Where it came from (file path, ``outer.zip:inner.zip``).
        task_id: CVAT task id.
        name: Task name (= clip ID).
        kind: ``video`` (tracks) or ``images`` (per-image shapes).
        updated: Task update time (last saved annotation).
        dumped: Export time.
        root: Parsed XML.
    """

    source: str
    task_id: str
    name: str
    kind: str
    updated: str
    dumped: str
    root: ET.Element


def iter_xml(path: Path) -> Iterator[tuple[str, bytes]]:
    """All ``annotations.xml`` in a file, a folder or (nested) zips.

    Args:
        path: XML file, folder or zip.

    Yields:
        ``(source description, xml bytes)``.
    """
    def from_zip(data: bytes, label: str) -> Iterator[tuple[str, bytes]]:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if info.filename.startswith("__MACOSX") or info.is_dir():
                    continue
                inner = zf.read(info)
                if info.filename.lower().endswith(".zip"):
                    yield from from_zip(inner, f"{label}:{info.filename}")
                elif info.filename.lower().endswith(".xml"):
                    yield f"{label}:{info.filename}", inner

    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.suffix.lower() in {".zip", ".xml"} and "__MACOSX" not in p.parts:
                yield from iter_xml(p)
    elif path.suffix.lower() == ".zip":
        yield from from_zip(path.read_bytes(), str(path))
    elif path.suffix.lower() == ".xml":
        yield str(path), path.read_bytes()


def parse_export(source: str, data: bytes) -> Export | None:
    """Reads the task meta of one export (None if it is not a CVAT 1.1 task export).

    Args:
        source: Where the XML came from.
        data: XML bytes.

    Returns:
        Parsed export or None.
    """
    root = ET.fromstring(data)
    task = root.find("meta/task")
    if root.tag != "annotations" or task is None:
        LOGGER.warning("Skipping %s: not a CVAT 1.1 task export", source)
        return None
    kind = "images" if root.find("image") is not None or "for images" in source else "video"
    return Export(source=source, task_id=task.findtext("id", ""), name=task.findtext("name", ""), kind=kind,
                  updated=task.findtext("updated", ""), dumped=root.findtext("meta/dumped", ""), root=root)


def latest(exports: list[Export]) -> dict[tuple[str, str], Export]:
    """Latest export per (task, kind) by task update time, then export time.

    Args:
        exports: All exports found.

    Returns:
        ``{(task_id, kind): export}``.
    """
    best: dict[tuple[str, str], Export] = {}
    for e in exports:
        key = (e.task_id, e.kind)
        if key not in best or (e.updated, e.dumped) > (best[key].updated, best[key].dumped):
            best[key] = e
    return best


def frame_map(images: Export | None, clip_id: str, processed: list[int], size: int) -> list[int]:
    """CVAT frame number -> clip frame index.

    Args:
        images: Images export of the task (frame file names), or None.
        clip_id: Clip of the task.
        processed: Processed frame indices of the clip, sorted (fallback).
        size: Number of frames in the task.

    Returns:
        ``frame_idx`` per CVAT frame.

    Raises:
        ValueError: If the names or the frame count do not match the clip.
    """
    if images is not None:
        out = [-1] * size
        for img in images.root.findall("image"):
            m = FRAME_NAME.match(Path(img.get("name", "")).name)
            if not m or m["clip"] != clip_id:
                raise ValueError(f"{images.source}: frame name {img.get('name')} is not {clip_id}_<idx>.jpg")
            out[int(img.get("id"))] = int(m["idx"])
        if -1 not in out:
            return out
        LOGGER.warning("%s: images export misses frames, using the processed frame order", images.source)
    if len(processed) != size:
        raise ValueError(f"{clip_id}: task has {size} frames, the clip has {len(processed)} processed frames")
    return processed


def frame_size(images: Export | None, video: Export, catalog_size: tuple[int, int] | None) -> tuple[int, int]:
    """Frame width and height: images export, else the frame catalog, else the video export meta.

    Args:
        images: Images export of the task, or None.
        video: Video export of the task.
        catalog_size: ``(width, height)`` from the frame catalog, or None.

    Returns:
        ``(width, height)``.

    Raises:
        ValueError: If no source knows the size.
    """
    img = images.root.find("image") if images is not None else None
    if img is not None:
        return int(img.get("width")), int(img.get("height"))
    if catalog_size is not None:
        return catalog_size
    w, h = video.root.findtext("meta/task/original_size/width"), video.root.findtext("meta/task/original_size/height")
    if w and h:
        return int(w), int(h)
    raise ValueError(f"{video.source}: frame size unknown (no images export, no frame catalog)")


def read_tracks(video: Export, fmap: list[int], keep: set[str]) -> pd.DataFrame:
    """Ground-truth boxes of one task: kept labels, inside the image.

    Args:
        video: Video export (tracks).
        fmap: CVAT frame -> clip frame index.
        keep: Labels to keep.

    Returns:
        One row per box.
    """
    rows: list[dict[str, Any]] = []
    for trk in video.root.findall("track"):
        if trk.get("label") not in keep:
            continue
        for b in trk.findall("box"):
            if b.get("outside") == "1":
                continue
            attrs = {a.get("name"): (a.text or "") for a in b.findall("attribute")}
            rows.append({"frame_idx": fmap[int(b.get("frame"))], "track_id": int(trk.get("id")),
                         "label": trk.get("label"), "x1": float(b.get("xtl")), "y1": float(b.get("ytl")),
                         "x2": float(b.get("xbr")), "y2": float(b.get("ybr")), "keyframe": b.get("keyframe") == "1",
                         "occluded": b.get("occluded") == "1", "source": trk.get("source", ""),
                         "tier": attrs.get("tier", "")})
    cols = ["frame_idx", "track_id", "label", "x1", "y1", "x2", "y2", "keyframe", "occluded", "source", "tier"]
    return pd.DataFrame(rows, columns=cols)


def merge_duplicates(gt: pd.DataFrame, iou_thr: float, contain_thr: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leaves one box per vehicle and frame; links fragments of one vehicle into one ``vehicle_id``.

    Review left two kinds of double labels (stacked boxes are hard to see in CVAT, but in the
    metrics the extra one is a "missed vehicle"):
        duplicate  two boxes on one vehicle, IoU > ``iou_thr`` (two auto tracks on one car,
                   or a hand-drawn track over auto fragments of it);
        inside     the box lies >= ``contain_thr`` inside another one: a truck cab inside the
                   whole-truck box;
        container  the box holds another one: a group box over parked cars, an oversized box.
    Which box stays is decided by persistence, not size: hand-drawn tracks first, then longer
    tracks. A real vehicle is seen for many frames; a cab fragment or a group box flickers.
    Fragments (duplicate, inside) join the vehicle of the kept box; a dropped container does not,
    because it may cover several vehicles.

    Args:
        gt: Ground-truth boxes of one clip.
        iou_thr: IoU above which two boxes are one vehicle.
        contain_thr: Share of a box inside another above which it is a part of it.

    Returns:
        Kept boxes with ``vehicle_id``; dropped boxes with ``reason`` and the kept box
        (``k_x1..k_y2``, ``k_track_id``).
    """
    if gt.empty:
        return gt.assign(vehicle_id=pd.Series(dtype=int)), pd.DataFrame()
    length = gt.groupby("track_id")["frame_idx"].transform("size")
    gt = gt.assign(_prio=(gt["source"] == "manual").astype(int) * 1_000_000 + length)
    parent = {t: t for t in gt["track_id"].unique()}

    def find(t: int) -> int:
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    dropped: list[dict[str, Any]] = []
    keep = np.ones(len(gt), dtype=bool)
    pos = {idx: i for i, idx in enumerate(gt.index)}
    for _, frame in gt.groupby("frame_idx"):
        frame = frame.sort_values("_prio", ascending=False)
        boxes = frame[["x1", "y1", "x2", "y2"]].to_numpy()
        tids = frame["track_id"].to_numpy()
        kept: list[int] = []
        for i in range(len(frame)):
            reason, j = "", -1
            if kept:
                kb = boxes[kept]
                ious = iou_one_to_many(boxes[i], kb)
                inside = np.array([contained_fraction(k, boxes[i][None])[0] for k in kb])  # box i inside kept k
                holds = contained_fraction(boxes[i], kb)                                   # kept k inside box i
                for r, score, thr in (("duplicate", ious, iou_thr), ("inside", inside, contain_thr),
                                      ("container", holds, contain_thr)):
                    if score.max() > thr:
                        reason, j = r, int(np.argmax(score))
                        break
            if not reason:
                kept.append(i)
                continue
            keep[pos[frame.index[i]]] = False
            k = kept[j]
            if reason != "container":
                parent[find(tids[i])] = find(tids[k])
            row = frame.iloc[i]
            dropped.append({**row.drop("_prio").to_dict(), "reason": reason, "k_track_id": int(tids[k]),
                            **dict(zip(("k_x1", "k_y1", "k_x2", "k_y2"), boxes[k]))})
    out = gt[keep].drop(columns="_prio")
    return out.assign(vehicle_id=out["track_id"].map(find)), pd.DataFrame(dropped)


def interpolate_frames(boxes: pd.DataFrame, step: int) -> pd.DataFrame:
    """Fills the frames between labelled frames of each vehicle by linear interpolation.

    Labels exist on every ``step``-th frame (the frames that were auto-labelled and reviewed).
    Between two labelled frames ``step`` apart where the same vehicle has a box, the frames in
    between get a linearly interpolated box. Nothing is added before a vehicle's first or after its
    last labelled frame, nor across a longer gap (it left the image or was hidden).

    Args:
        boxes: Boxes of one clip with ``frame_idx, vehicle_id, track_id, x1..y2`` (one per vehicle and frame).
        step: Frame step of the labelled frames.

    Returns:
        ``boxes`` plus the new rows, ``observed`` = True on labelled frames.
    """
    out = boxes.assign(observed=True)
    if step <= 1 or boxes.empty:
        return out
    coords = ["x1", "y1", "x2", "y2"]
    new = []
    for _, v in boxes.sort_values("frame_idx").groupby("vehicle_id"):
        v = v.drop_duplicates("frame_idx")
        f = v["frame_idx"].to_numpy()
        xy = v[coords].to_numpy(dtype=float)
        for i in np.flatnonzero(np.diff(f) == step):
            for j in range(1, step):
                w = j / step
                row = v.iloc[i].to_dict()
                row.update(dict(zip(coords, (1 - w) * xy[i] + w * xy[i + 1])), frame_idx=int(f[i] + j),
                           keyframe=False, observed=False)
                new.append(row)
    return pd.concat([out, pd.DataFrame(new)], ignore_index=True) if new else out


def render_dropped(dropped: pd.DataFrame, paths: dict[tuple[str, int], str], datasets_root: Path, out: Path,
                   rng: np.random.Generator, per_reason: int = 16, thumb: int = 200) -> None:
    """Crop sheet of dropped double boxes: green = kept box, red = dropped; one row block per reason.

    Args:
        dropped: Dropped boxes (``merge_duplicates``) with ``clip_id``.
        paths: ``(clip_id, frame_idx) -> frame path``.
        datasets_root: Root for frame paths.
        out: Output JPEG.
        rng: Random generator (which examples are shown).
        per_reason: Examples per reason.
        thumb: Tile size, px.
    """
    cols = 8
    picks = [g.iloc[rng.permutation(len(g))[:per_reason]] for _, g in dropped.groupby("reason", sort=True)]
    rows = sum((len(p) + cols - 1) // cols for p in picks)
    sheet = np.full((rows * thumb, cols * thumb, 3), 255, np.uint8)
    r0 = 0
    for part in picks:
        for n, d in enumerate(part.itertuples(index=False)):
            img = cv2.imread(str(datasets_root / paths[(d.clip_id, d.frame_idx)]))
            ux1, uy1 = min(d.x1, d.k_x1), min(d.y1, d.k_y1)
            ux2, uy2 = max(d.x2, d.k_x2), max(d.y2, d.k_y2)
            side = int(max(1.6 * max(ux2 - ux1, uy2 - uy1), 96))
            x0 = int(np.clip((ux1 + ux2 - side) / 2, 0, max(img.shape[1] - side, 0)))
            y0 = int(np.clip((uy1 + uy2 - side) / 2, 0, max(img.shape[0] - side, 0)))
            crop = img[y0:y0 + side, x0:x0 + side]
            f = thumb / max(crop.shape[:2])
            tile = np.full((thumb, thumb, 3), 255, np.uint8)
            small = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            tile[:small.shape[0], :small.shape[1]] = small
            for (a1, b1, a2, b2), color, w in (((d.k_x1, d.k_y1, d.k_x2, d.k_y2), (60, 200, 60), 2),
                                                ((d.x1, d.y1, d.x2, d.y2), (40, 40, 230), 1)):
                cv2.rectangle(tile, (int((a1 - x0) * f), int((b1 - y0) * f)), (int((a2 - x0) * f), int((b2 - y0) * f)),
                              color, w)
            text = f"{d.reason} {d.clip_id}"
            cv2.putText(tile, text, (3, thumb - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
            cv2.putText(tile, text, (3, thumb - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            r, c = divmod(n, cols)
            sheet[(r0 + r) * thumb:(r0 + r + 1) * thumb, c * thumb:(c + 1) * thumb] = tile
        r0 += (len(part) + cols - 1) // cols
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def render_gt_videos(out_dir: Path, frames: pd.DataFrame, gt: pd.DataFrame, dropped: pd.DataFrame,
                     datasets_root: Path) -> None:
    """Whole-clip videos of the final ground truth, one per clip.

    Green: boxes from the auto labels the reviewer kept; cyan: drawn by hand; thin red: double
    boxes removed here (``dup`` / ``part`` / ``group``). Label = vehicle id.

    Args:
        out_dir: Folder for ``<clip>_gt.mp4``.
        frames: Processed frames of the clips (``clip_id, frame_idx, t_s, path``).
        gt: Ground-truth boxes.
        dropped: Dropped double boxes (may be empty).
        datasets_root: Root for frame paths.
    """
    short = {"duplicate": "dup", "inside": "part", "container": "group"}
    for clip_id, fr in frames.groupby("clip_id", sort=True):
        g = gt[gt["clip_id"] == clip_id].assign(kind=lambda d: np.where(d["source"] == "manual", "manual", "auto"))
        d = (dropped[dropped["clip_id"] == clip_id].assign(kind="dropped", vehicle_id=-1)
             if len(dropped) else pd.DataFrame(columns=g.columns))
        boxes = pd.concat([d, g], ignore_index=True)  # dropped first, ground truth drawn on top
        legend = [(f"GT from auto labels: {g.loc[g['kind'] == 'auto', 'vehicle_id'].nunique()} vehicles", (60, 200, 60)),
                  (f"GT drawn by hand: {g.loc[g['kind'] == 'manual', 'vehicle_id'].nunique()} vehicles", (255, 200, 0)),
                  (f"removed double boxes: {len(d)}", (40, 40, 230))]

        def style(r: Any) -> tuple[tuple[int, int, int], str, int]:
            if r.kind == "dropped":
                return (40, 40, 230), short.get(r.reason, ""), 1
            return ((255, 200, 0) if r.kind == "manual" else (60, 200, 60)), str(int(r.vehicle_id)), 2

        render_boxes_video(out_dir / f"{clip_id}_gt.mp4", fr.sort_values("frame_idx"), boxes, datasets_root, 0, style,
                           legend)


def review_report(gt: pd.DataFrame, pre: pd.DataFrame | None) -> dict[str, Any]:
    """What review changed for one clip.

    A ground-truth track is "auto" if CVAT says it came from the uploaded file; its tier is the
    stage 05 ``tier`` attribute it still carries (empty if the reviewer changed the label, which
    resets attributes). Survival of a tier = auto GT tracks of that tier / pre-label tracks of it.

    Args:
        gt: Ground-truth boxes of the clip.
        pre: Stage 05 track table of the clip (tier per pre-label track), or None.

    Returns:
        Report dictionary.
    """
    tr = gt.groupby("track_id").agg(source=("source", "first"),
                                    tier=("tier", lambda t: t[t != ""].mode().iat[0] if (t != "").any() else ""))
    out: dict[str, Any] = {"gt_tracks": len(tr), "gt_boxes": len(gt),
                           "manual_tracks": int((tr["source"] == "manual").sum()),
                           "auto_tracks_by_tier": tr.loc[tr["source"] != "manual", "tier"].replace("", "relabelled")
                                                    .value_counts().to_dict()}
    if pre is not None:
        pre_n = pre["tier"].value_counts()
        out["pre_label_tracks_by_tier"] = pre_n.to_dict()
        out["survival_by_tier"] = {t: round(out["auto_tracks_by_tier"].get(t, 0) / n, 3)
                                   for t, n in pre_n.items() if t in ("A_sure", "B_likely", "C_rescued") and n}
    return out


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    parser = base_arg_parser(__doc__)
    parser.add_argument("--exports", type=Path, nargs="+", default=None,
                        help="Export zips / folders / outer zip (default: import.exports_dir).")
    parser.add_argument("--no-video", action="store_true", help="Skip the ground-truth videos.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Write the ground truth even if some eval clips have no export.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    stage = start_stage("06_import", cfg, args)
    icfg = cfg["import"]
    camera, clips = load_clips(Path(cfg["paths"]["clips"]))
    roles = {c.clip_id: c.role for c in clips}
    edges = [float(e) for e in cfg["fuse"]["bands_m"]]

    exports = [e for src in (args.exports or [Path(icfg["exports_dir"])]) for s, data in iter_xml(Path(src).expanduser())
               if (e := parse_export(s, data)) is not None]
    chosen = latest(exports)
    used = {id(e) for e in chosen.values()}
    pd.DataFrame([{"source": e.source, "task_id": e.task_id, "clip_id": e.name, "kind": e.kind, "updated": e.updated,
                   "dumped": e.dumped, "used": id(e) in used} for e in exports]).to_csv(stage.out_dir / "exports.csv",
                                                                                      index=False)
    LOGGER.info("%d exports found, %d used (latest per task and kind)", len(exports), len(used))
    if not exports:  # never overwrite the existing ground truth with an empty one
        LOGGER.error("No CVAT exports in %s: put the export zips there or pass --exports", args.exports or
                     icfg["exports_dir"])
        return 1

    catalog_path = Path(cfg["paths"]["catalog"])
    processed: dict[str, list[int]] = {}
    paths: dict[tuple[str, int], str] = {}
    sizes: dict[str, tuple[int, int]] = {}
    frames = None
    if catalog_path.is_file():
        frames = select_frames(pd.read_parquet(catalog_path), frame_steps(clips, int(cfg["predict"]["train_frame_step"])),
                               None)
        processed = {c: g["frame_idx"].tolist() for c, g in frames.groupby("clip_id")}
        sizes = {c: (int(g["width"].iloc[0]), int(g["height"].iloc[0])) for c, g in frames.groupby("clip_id")}
        paths = dict(zip(zip(frames["clip_id"], frames["frame_idx"]), frames["path"]))
    triage_path = Path(icfg.get("triage_tracks") or "")
    triage = pd.read_parquet(triage_path) if triage_path.is_file() else None
    keep = set(icfg["keep_labels"])

    parts, stats, dropped_parts = [], {}, []
    for task_id in sorted({k[0] for k in chosen}, key=int):
        video = chosen.get((task_id, "video"))
        if video is None:
            LOGGER.warning("Task %s: no 'CVAT for video 1.1' export, skipped", task_id)
            continue
        clip_id = video.name
        if roles.get(clip_id) not in set(icfg["roles"]):
            LOGGER.warning("Task %s: '%s' is not a clip with role %s, skipped", task_id, clip_id, icfg["roles"])
            continue
        size = int(video.root.findtext("meta/task/size"))
        fmap = frame_map(chosen.get((task_id, "images")), clip_id, processed.get(clip_id, []), size)
        raw = read_tracks(video, fmap, keep)  # review report on what the reviewer left, before deduplication
        gt, dropped = merge_duplicates(raw, float(icfg["duplicate_iou"]), float(icfg["part_inside"]))
        n_dup = len(dropped)
        if len(dropped):
            dropped_parts.append(dropped.assign(clip_id=clip_id))
        width, height = frame_size(chosen.get((task_id, "images")), video, sizes.get(clip_id))
        f_px = focal_px(width, height, float(camera["hfov_deg"]))
        steps_here = frame_steps(clips, int(cfg["predict"]["train_frame_step"]))
        n_obs = len(gt)
        gt = interpolate_frames(gt, int(steps_here.get(clip_id, 1)))  # metrics are per frame
        gt.insert(0, "clip_id", clip_id)
        gt["dist_m"] = distance_m(gt[["x1", "y1", "x2", "y2"]].to_numpy(), f_px, float(camera["ref_length_m"])) \
            if len(gt) else np.zeros(0)
        gt["band"] = band_labels(gt["dist_m"].to_numpy(), edges) if len(gt) else []
        parts.append(gt)
        pre = triage[triage["clip_id"] == clip_id] if triage is not None else None
        stats[clip_id] = {"task_id": task_id, "export": video.source, "frames": size,
                          "double_boxes_dropped": dropped["reason"].value_counts().to_dict() if n_dup else {},
                          "labelled_boxes": n_obs, "boxes_all_frames": len(gt), "vehicles": int(gt["vehicle_id"].nunique()) if len(gt) else 0,
                          **(review_report(raw, pre) if len(raw) else {"gt_tracks": 0, "gt_boxes": 0}),
                          "gt_vehicles_by_band": vehicle_bands(gt, ["vehicle_id"]).value_counts().sort_index().to_dict()
                          if len(gt) else {}}
        LOGGER.info("%s: %d vehicles (%d tracks after review, %d added by hand), %d boxes (%d double boxes dropped), "
                    "by band %s", clip_id, stats[clip_id]["vehicles"], stats[clip_id]["gt_tracks"],
                    stats[clip_id].get("manual_tracks", 0), len(gt), n_dup, stats[clip_id]["gt_vehicles_by_band"])

    gt_all = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    missing = sorted(c for c, r in roles.items() if r in set(icfg["roles"]) and c not in stats)
    if not stats or (missing and not args.allow_partial):  # never replace the ground truth with a partial one
        LOGGER.error("No usable export for %s: %s not written (fix the exports or pass --allow-partial)",
                     missing or "any clip", icfg["gt_path"])
        return 1
    if missing:
        LOGGER.warning("No export for %s: the ground truth has no boxes for them (--allow-partial)", missing)
    out_path = Path(icfg["gt_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gt_all.to_parquet(out_path, index=False)
    dropped_all = pd.DataFrame()
    if dropped_parts:
        dropped_all = pd.concat(dropped_parts, ignore_index=True)
        dropped_all.to_parquet(stage.out_dir / "dropped.parquet", index=False)
        if paths:
            render_dropped(dropped_all, paths, Path(cfg["paths"]["datasets"]), stage.viz_dir / "dropped.jpg",
                           np.random.default_rng(int(cfg["seed"])))
    vehicles = vehicle_bands(gt_all, ["clip_id", "vehicle_id"]) if len(gt_all) else pd.Series(dtype=str)
    stats["_total"] = {"clips": len(parts), "vehicles": int(len(vehicles)),
                       "gt_tracks": int(gt_all.groupby(["clip_id", "track_id"]).ngroups) if len(gt_all) else 0,
                       "gt_boxes": int(len(gt_all)),
                       "gt_vehicles_by_band": vehicles.value_counts().sort_index().to_dict(),
                       "missing_clips": missing}
    review = pd.DataFrame({c: v.get("auto_tracks_by_tier", {}) for c, v in stats.items() if not c.startswith("_")}).T
    if triage is not None and len(review):
        pre = triage[triage["clip_id"].isin(review.index)].groupby("tier").size()
        kept = review.sum()
        stats["_total"]["pre_label_survival"] = {t: {"pre": int(pre.get(t, 0)), "kept": int(kept.get(t, 0)),
                                                     "share": round(float(kept.get(t, 0) / pre[t]), 3)}
                                                 for t in ("A_sure", "B_likely", "C_rescued") if pre.get(t, 0)}
        LOGGER.info("Pre-label tracks kept by the reviewer, per tier: %s", stats["_total"]["pre_label_survival"])
    stats["_total"]["manual_tracks"] = int(sum(v.get("manual_tracks", 0) for c, v in stats.items()
                                               if not c.startswith("_")))
    stage.save_stats(stats)
    LOGGER.info("Ground truth: %s -> %s", stats["_total"], out_path)
    if icfg.get("video", True) and not args.no_video:
        if frames is None:
            LOGGER.warning("No frame catalog: ground-truth videos skipped")
        else:
            clips_done = [c for c in stats if not c.startswith("_")]
            render_gt_videos(stage.viz_dir, frames[frames["clip_id"].isin(clips_done)], gt_all, dropped_all,
                             Path(cfg["paths"]["datasets"]))
            LOGGER.info("Ground-truth videos (green auto, cyan hand-drawn, thin red dropped doubles) -> %s",
                        stage.viz_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
