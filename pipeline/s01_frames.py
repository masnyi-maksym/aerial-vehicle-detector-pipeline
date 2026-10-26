"""Stage 01: decode every clip once into JPEG frames and build the frame catalog.

Frames keep native resolution and orientation. Each frame is identified by
``(clip_id, frame_idx)``; ``frame_idx`` starts at 0 in decode order, so
``t = frame_idx / fps`` is exact for these constant-frame-rate clips.

Outputs:
    datasets/frames/{clip_id}/{clip_id}_{idx:06d}.jpg
    datasets/meta/frames.parquet          one row per frame
    outputs/01_frames/{config.yaml, log.txt, stats.json}

Example:
    python -m pipeline.s01_frames --config configs/pipeline.yaml
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.common import Clip, base_arg_parser, load_clips, load_yaml, start_stage
from pipeline.geometry import box_px_at, focal_px

LOGGER = logging.getLogger("pipeline.s01_frames")


def probe(video: Path) -> dict[str, Any]:
    """Reads stream properties and counts frames by decoding (``nb_frames`` can be N/A).

    Args:
        video: Video file.

    Returns:
        Mapping with ``width``, ``height``, ``fps`` (Fraction), ``n_frames``.
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames",
        "-of", "json", str(video),
    ]
    stream = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)["streams"][0]
    if stream["r_frame_rate"] != stream["avg_frame_rate"]:
        LOGGER.warning("%s: variable frame rate (%s vs %s), t = idx / fps is approximate",
                       video.name, stream["r_frame_rate"], stream["avg_frame_rate"])
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": Fraction(stream["r_frame_rate"]),
        "n_frames": int(stream["nb_read_frames"]),
    }


def extract(video: Path, out_dir: Path, clip_id: str, qscale: int) -> int:
    """Decodes all frames to JPEG with ffmpeg, numbering from 0.

    ``-fps_mode passthrough`` stops ffmpeg from duplicating or dropping frames
    to match a nominal rate; ``-start_number 0`` aligns file numbers with
    ``frame_idx``.

    Args:
        video: Source video.
        out_dir: Destination folder (emptied first).
        clip_id: Prefix of the frame file names.
        qscale: ffmpeg ``-q:v`` value for JPEG quality (2 = best).

    Returns:
        Number of written frames.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-map", "0:v:0",
        "-fps_mode", "passthrough", "-start_number", "0", "-q:v", str(qscale),
        str(out_dir / f"{clip_id}_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob("*.jpg")))


def process_clip(
    clip: Clip, camera: dict[str, Any], datasets_root: Path, frames_root: Path, cfg: dict[str, Any], overwrite: bool
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extracts one clip and returns its catalog rows and stats.

    Args:
        clip: Clip to process.
        camera: Camera assumptions from ``clips.yaml``.
        datasets_root: Root that ``clip.file`` is relative to.
        frames_root: Root folder for extracted frames.
        cfg: ``frames`` section of the pipeline config.
        overwrite: Re-extract even if the frame count already matches.

    Returns:
        Catalog rows of the clip and its stats.

    Raises:
        FileNotFoundError: If the video is missing.
        RuntimeError: If the number of extracted frames differs from ffprobe.
    """
    video = datasets_root / clip.file
    if not video.is_file():
        raise FileNotFoundError(f"{clip.clip_id}: video not found: {video} (source: {clip.url})")

    info = probe(video)
    out_dir = frames_root / clip.clip_id
    existing = len(list(out_dir.glob("*.jpg"))) if out_dir.exists() else 0
    if overwrite or existing != info["n_frames"]:
        n_written = extract(video, out_dir, clip.clip_id, int(cfg["jpeg_qscale"]))
    else:
        LOGGER.info("%s: %d frames already extracted, skipping (use --overwrite)", clip.clip_id, existing)
        n_written = existing
    if n_written != info["n_frames"]:
        raise RuntimeError(f"{clip.clip_id}: extracted {n_written} frames, ffprobe counted {info['n_frames']}")

    paths = [out_dir / f"{clip.clip_id}_{i:06d}.jpg" for i in range(n_written)]
    fps = info["fps"]
    f_px = focal_px(info["width"], info["height"], float(camera["hfov_deg"]))
    ref_len = float(camera["ref_length_m"])

    rows = pd.DataFrame({
        "clip_id": clip.clip_id,
        "role": clip.role,
        "frame_idx": np.arange(n_written),
        "t_s": [float(i / fps) for i in range(n_written)],
        "width": info["width"],
        "height": info["height"],
        "fps": float(fps),
        "f_px": f_px,
        "path": [p.relative_to(datasets_root).as_posix() for p in paths],
    })
    stats = {
        "role": clip.role,
        "slot": clip.slot,
        "file": clip.file,
        "resolution": f"{info['width']}x{info['height']}",
        "fps": f"{fps.numerator}/{fps.denominator}",
        "n_frames": n_written,
        "duration_s": round(float(n_written / fps), 3),
        "f_px": round(f_px, 1),
        "car_px_at_200m": round(box_px_at(200, f_px, ref_len), 1),
        "car_px_at_400m": round(box_px_at(400, f_px, ref_len), 1),
    }
    return rows, stats


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    parser = base_arg_parser(__doc__)
    parser.add_argument("--clips", nargs="*", default=None, help="Only these clip IDs (default: all).")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract frames even if counts match.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    stage = start_stage("01_frames", cfg, args)
    paths = cfg["paths"]
    datasets_root, frames_root, catalog_path = Path(paths["datasets"]), Path(paths["frames"]), Path(paths["catalog"])
    camera, clips = load_clips(Path(paths["clips"]))
    if args.clips:
        clips = [c for c in clips if c.clip_id in set(args.clips)]

    tables, stats = [], {}
    for clip in clips:
        rows, clip_stats = process_clip(clip, camera, datasets_root, frames_root, cfg["frames"], args.overwrite)
        tables.append(rows)
        stats[clip.clip_id] = clip_stats
        LOGGER.info("%s %-5s %s @ %s fps: %d frames, car at 200/400 m = %.0f/%.0f px",
                    clip.clip_id, clip.role, clip_stats["resolution"], clip_stats["fps"], clip_stats["n_frames"],
                    clip_stats["car_px_at_200m"], clip_stats["car_px_at_400m"])

    catalog = pd.concat(tables, ignore_index=True)
    if args.clips and catalog_path.exists():  # partial run: replace only the processed clips
        old = pd.read_parquet(catalog_path)
        catalog = pd.concat([old[~old["clip_id"].isin(catalog["clip_id"])], catalog], ignore_index=True)
    catalog = catalog.sort_values(["clip_id", "frame_idx"], ignore_index=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(catalog_path, index=False)
    LOGGER.info("Catalog: %s (%d frames)", catalog_path, len(catalog))

    stage.save_stats({"clips": stats, "total_frames": int(len(catalog)),
                      "frames_by_role": catalog.groupby("role").size().to_dict()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
