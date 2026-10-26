"""Shared helpers: config and clip registry loading, seeding, per-stage output folders."""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger("pipeline")
ROLES = ("train", "val", "eval")


@dataclass(frozen=True)
class Clip:
    """One source video from the clip registry.

    Attributes:
        clip_id: Pexels page ID, the key used everywhere in the pipeline.
        role: ``train``, ``val`` (whole held-out clip: picks the frame step / checkpoint),
            or ``eval`` (reviewed, used for metrics only).
        slot: Slot name from the task (A-D or eval).
        scene: Short scene description.
        url: Pexels page URL.
        file: Video path relative to the datasets root.
        frame_step: Optional per-clip frame step (default: train_frame_step for train / val, else 1).
    """

    clip_id: str
    role: str
    slot: str
    scene: str
    url: str
    file: str
    frame_step: int | None = None


def load_yaml(path: Path) -> dict[str, Any]:
    """Reads a YAML file.

    Args:
        path: File to read.

    Returns:
        Parsed mapping.
    """
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_clips(path: Path) -> tuple[dict[str, Any], list[Clip]]:
    """Loads the clip registry.

    Args:
        path: Path to ``clips.yaml``.

    Returns:
        Camera assumptions and the list of clips.

    Raises:
        ValueError: On duplicate clip IDs or an unknown role.
    """
    raw = load_yaml(path)
    clips = [Clip(**{k: (int(v) if k == "frame_step" else str(v)) for k, v in c.items()}) for c in raw["clips"]]
    ids = [c.clip_id for c in clips]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate clip_id in {path}: {ids}")
    for c in clips:
        if c.role not in ROLES:
            raise ValueError(f"Clip {c.clip_id}: unknown role '{c.role}'")
    return raw["camera"], clips


def frame_steps(clips: list[Clip], train_step: int) -> dict[str, int]:
    """Frame step per clip: explicit ``frame_step``, else ``train_step`` for train / val and 1 otherwise.

    Steps keep ``frame_idx % step == 0``, so a coarser step is always a subset of a finer one.

    Args:
        clips: Clip registry.
        train_step: Default step of train clips.

    Returns:
        ``{clip_id: step}``.
    """
    return {c.clip_id: c.frame_step or (train_step if c.role in ("train", "val") else 1) for c in clips}


def vehicle_bands(boxes: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Distance band of each vehicle: the band of its first box in time (where it first appears).

    The one definition used for every per-vehicle number (vehicles per band, time to first
    detection); detection rate and precision are per box and use each box's own band.

    Args:
        boxes: Boxes with ``frame_idx``, ``band`` and the ``keys`` columns.
        keys: Vehicle key columns, e.g. ``["clip_id", "vehicle_id"]``.

    Returns:
        Band per vehicle, indexed by ``keys``.
    """
    return boxes.sort_values("frame_idx", kind="stable").groupby(keys)["band"].first()


def set_seed(seed: int) -> None:
    """Seeds python, numpy and (if installed) torch; disables cuDNN autotuning.

    Code that needs randomness should still create its own generator, e.g.
    ``np.random.default_rng(seed)``, so adding a random call in one place
    never shifts sampling in another.

    Args:
        seed: Global seed from the config.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


@dataclass
class Stage:
    """Output folder of one pipeline stage: ``outputs/<name>/``.

    Attributes:
        name: Stage folder name, e.g. ``01_frames``.
        out_dir: Folder for config, log, stats, data and viz.
        cfg: Full pipeline config.
    """

    name: str
    out_dir: Path
    cfg: dict[str, Any]

    @property
    def viz_dir(self) -> Path:
        """Folder for sample visualisations (created on first access)."""
        path = self.out_dir / "viz"
        path.mkdir(exist_ok=True)
        return path

    def save_stats(self, stats: dict[str, Any]) -> Path:
        """Writes ``stats.json`` for the stage.

        Args:
            stats: JSON-serialisable summary numbers.

        Returns:
            Path to the written file.
        """
        path = self.out_dir / "stats.json"
        path.write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        LOGGER.info("Stats: %s", path)
        return path


def start_stage(name: str, cfg: dict[str, Any], args: argparse.Namespace) -> Stage:
    """Creates the stage folder, sets up logging, saves the config and seeds RNGs.

    Re-running a stage overwrites its files in place.

    Args:
        name: Stage folder name, e.g. ``02_predict``.
        cfg: Full pipeline config.
        args: Parsed CLI arguments (saved next to the config for reproducibility).

    Returns:
        Stage handle.
    """
    out_dir = Path(cfg["paths"]["outputs"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)  # third-party libraries (httpx, HF hub ...) only report warnings
    logging.getLogger("pipeline").setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "log.txt", mode="w")):
        handler.setFormatter(fmt)
        root.addHandler(handler)

    snapshot = {"cli": {k: str(v) for k, v in vars(args).items()}, **cfg}
    (out_dir / "config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8")

    set_seed(int(cfg["seed"]))
    LOGGER.info("Stage %s -> %s (seed=%s)", name, out_dir, cfg["seed"])
    return Stage(name=name, out_dir=out_dir, cfg=cfg)


class VideoSink:
    """Writes BGR frames to an H.264 mp4 through ffmpeg (plays in browsers, VLC, QuickTime).

    OpenCV's ``mp4v`` writes MPEG-4 Part 2, which many players show as a green screen.
    Frames are cropped to even width / height (yuv420p requirement).

    Example:
        with VideoSink(path, fps=10) as sink:
            for img in images:
                sink.write(img)
    """

    def __init__(self, path: Path, fps: float, crf: int = 23) -> None:
        """Prepares the sink; ffmpeg starts on the first frame (size is known then).

        Args:
            path: Output ``.mp4`` file.
            fps: Playback frame rate.
            crf: x264 quality (lower = better, 18-28 is the usual range).
        """
        self.path, self.fps, self.crf = path, fps, crf
        self._proc: subprocess.Popen[bytes] | None = None
        self._size: tuple[int, int] | None = None

    def write(self, img: np.ndarray) -> None:
        """Appends one BGR frame (all frames must have the size of the first one).

        Args:
            img: ``(H, W, 3)`` uint8 BGR image.
        """
        if self._proc is None:
            h, w = img.shape[:2]
            self._size = (w - w % 2, h - h % 2)
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{self._size[0]}x{self._size[1]}", "-r", f"{self.fps:.3f}", "-i", "-",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", str(self.crf),
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)]
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        w, h = self._size
        self._proc.stdin.write(np.ascontiguousarray(img[:h, :w]).tobytes())

    def close(self) -> None:
        """Finishes the file.

        Raises:
            RuntimeError: If ffmpeg failed.
        """
        if self._proc is None:
            return
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")
        self._proc = None

    def __enter__(self) -> VideoSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser with the options shared by all stages.

    Args:
        description: Help text of the stage.

    Returns:
        Parser with ``--config``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"), help="Pipeline config YAML.")
    return parser
