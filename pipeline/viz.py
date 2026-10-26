"""Shared visualisations: per-clip box videos and track crop sheets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

from pipeline.common import VideoSink

Style = Callable[[Any], tuple]  # box row -> (BGR colour, label text[, line thickness])


def draw_boxes(path: Path, boxes: pd.DataFrame | None, style: Style,
               legend: list[tuple[str, tuple[int, int, int]]] | None = None, max_side: int = 1920) -> np.ndarray:
    """Reads a frame, downscales it to ``max_side`` and draws boxes and a legend.

    Args:
        path: Frame image.
        boxes: Boxes of this frame (``x1..y2`` in frame pixels), drawn in row order; None = none.
        style: Box row (namedtuple) -> (BGR colour, label text[, thickness]).
        legend: Optional ``(text, BGR colour)`` lines in the top-left corner.
        max_side: Long side of the output image.

    Returns:
        BGR image.
    """
    img = cv2.imread(str(path))
    scale = min(1.0, max_side / max(img.shape[:2]))
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for r in (boxes.itertuples(index=False) if boxes is not None else []):
        st = style(r)
        color, label, thick = st[0], st[1], (st[2] if len(st) > 2 else 2)
        p1, p2 = (int(r.x1 * scale), int(r.y1 * scale)), (int(r.x2 * scale), int(r.y2 * scale))
        cv2.rectangle(img, p1, p2, color, thick)
        if label:
            cv2.putText(img, label, (p1[0], max(p1[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for i, (text, color) in enumerate(legend or []):
        y = 30 + 28 * i
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img


def render_boxes_video(path: Path, frames: pd.DataFrame, boxes: pd.DataFrame, datasets_root: Path, seconds: float,
                       style: Style, legend: list[tuple[str, tuple[int, int, int]]] | None = None) -> None:
    """Writes a clip (downscaled to 1920 px) with boxes drawn in a caller-defined style.

    Args:
        path: Output ``.mp4``.
        frames: Frames of the clip, sorted (``frame_idx, t_s, path``).
        boxes: Boxes of the clip (need ``frame_idx, x1..y2``).
        datasets_root: Root for catalog paths.
        seconds: Length from the clip start; ``<= 0`` renders the whole clip.
        style: Box row (namedtuple) -> (BGR colour, label text[, thickness]); boxes are drawn in row order.
        legend: Optional ``(text, BGR colour)`` lines drawn in the top-left corner of every frame.
    """
    sel = frames if seconds <= 0 else frames[frames["t_s"] <= frames["t_s"].iloc[0] + seconds]
    if len(sel) < 2:
        return
    fps = max(1.0, (len(sel) - 1) / max(sel["t_s"].iloc[-1] - sel["t_s"].iloc[0], 1e-6))
    by_frame = {k: g for k, g in boxes.groupby("frame_idx")}
    with VideoSink(path, fps) as sink:
        for fr in sel.itertuples(index=False):
            sink.write(draw_boxes(datasets_root / fr.path, by_frame.get(fr.frame_idx), style, legend))


def crop_sheet(groups: list[tuple[str, pd.DataFrame]], obs: pd.DataFrame, datasets_root: Path, path: Path,
               cols: int, caption: Callable[[Any], str], thumb: int = 128, font: float = 0.38) -> None:
    """Rows of track crops (middle observation, 3x context) with a caption per crop.

    Args:
        groups: ``(row title, tracks)`` per row; at most ``cols`` tracks (need ``clip_id, track_id``) per row.
        obs: Observed boxes (``clip_id, track_id, frame_idx, path, x1..y2``).
        datasets_root: Root for catalog paths.
        path: Output JPEG.
        cols: Crops per row.
        caption: Track row (namedtuple) -> caption text.
        thumb: Crop size, px.
        font: Caption font scale.
    """
    label_w = 150
    sheet = np.full((max(len(groups), 1) * (thumb + 4), label_w + cols * thumb, 3), 255, np.uint8)
    by_track = obs.groupby(["clip_id", "track_id"])
    for r, (title, tracks) in enumerate(groups):
        y0 = r * (thumb + 4)
        for i, line in enumerate(title.split("\n")):
            cv2.putText(sheet, line, (4, y0 + 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        for c, t in enumerate(tracks.head(cols).itertuples(index=False)):
            g = by_track.get_group((t.clip_id, t.track_id)).sort_values("frame_idx")
            b = g.iloc[len(g) // 2]
            img = cv2.imread(str(datasets_root / b["path"]))
            side = int(max(3 * max(b.x2 - b.x1, b.y2 - b.y1), 64))
            cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
            x0 = int(np.clip(cx - side / 2, 0, max(img.shape[1] - side, 0)))
            y0c = int(np.clip(cy - side / 2, 0, max(img.shape[0] - side, 0)))
            crop = img[y0c:y0c + side, x0:x0 + side]
            f = thumb / max(crop.shape[:2])
            tile = np.full((thumb, thumb, 3), 255, np.uint8)
            small = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            tile[:small.shape[0], :small.shape[1]] = small
            cv2.rectangle(tile, (int((b.x1 - x0) * f), int((b.y1 - y0c) * f)),
                          (int((b.x2 - x0) * f), int((b.y2 - y0c) * f)), (0, 255, 255), 1)
            text = caption(t)
            cv2.putText(tile, text, (2, thumb - 5), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), 3)
            cv2.putText(tile, text, (2, thumb - 5), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), 1)
            sheet[y0:y0 + thumb, label_w + c * thumb:label_w + (c + 1) * thumb] = tile
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
