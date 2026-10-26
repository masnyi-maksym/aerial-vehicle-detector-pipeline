"""Thin wrappers that give every labeler the same interface: BGR images in, boxes out.

Each wrapper runs at its native input size (``tile``), so tiles of that size
reach the network without rescaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np


@dataclass
class Detections:
    """Detections of one image.

    Attributes:
        boxes: ``(N, 4)`` float32 xyxy boxes in input-image pixels.
        scores: ``(N,)`` float32 confidences.
        labels: ``N`` source class names (``car``, ``truck`` ...).
    """

    boxes: np.ndarray
    scores: np.ndarray
    labels: list[str]


class Detector(Protocol):
    """Common detector interface."""

    name: str
    tile: int
    batch: int

    def predict(self, images: list[np.ndarray]) -> list[Detections]:
        """Runs inference on BGR images (one batch)."""
        ...


def is_target(label: str, targets: list[str]) -> bool:
    """Whether a Grounding DINO label names a target class (empty label = unknown = yes).

    Args:
        label: Text label of a box, e.g. ``"car"`` or ``"traffic cone"``.
        targets: Target class names from the prompt.

    Returns:
        True for target or unknown labels.
    """
    return not label or any(t in label for t in targets)


def _torch_device(device: int | str) -> str:
    """Converts a config device (``0`` or ``"cpu"``) to a torch device string."""
    return f"cuda:{device}" if str(device).isdigit() else str(device)


class UltralyticsDetector:
    """YOLO or RT-DETR from Ultralytics, restricted to vehicle classes."""

    def __init__(self, name: str, cfg: dict[str, Any], device: int | str) -> None:
        """Loads the model.

        Args:
            name: Model key from the config.
            cfg: Model config section.
            device: CUDA index or ``"cpu"``.
        """
        from ultralytics import RTDETR, YOLO

        self.name = name
        self.tile = int(cfg["tile"])
        self.batch = int(cfg["batch"])
        self.conf = float(cfg["conf"])
        self.classes = [int(c) for c in cfg["classes"]]
        self.quantize = 32 if str(device) == "cpu" else int(cfg.get("quantize", 32))  # 16 = FP16 compute
        self.device = device
        self.model = RTDETR(cfg["weights"]) if cfg["type"] == "rtdetr" else YOLO(cfg["weights"])

    def predict(self, images: list[np.ndarray]) -> list[Detections]:
        """Runs inference; non-square images (full-frame pass) are letterboxed to ``tile``.

        Args:
            images: BGR images.

        Returns:
            One ``Detections`` per image.
        """
        results = self.model.predict(
            source=images, imgsz=self.tile, conf=self.conf, classes=self.classes,
            quantize=self.quantize, device=self.device, verbose=False,
        )
        out = []
        for res in results:
            cls = res.boxes.cls.cpu().numpy().astype(int)
            out.append(Detections(
                boxes=res.boxes.xyxy.cpu().numpy().astype(np.float32),
                scores=res.boxes.conf.cpu().numpy().astype(np.float32),
                labels=[res.names[c] for c in cls],
            ))
        return out


class GroundingDinoDetector:
    """Grounding DINO (Hugging Face transformers) with a text prompt."""

    def __init__(self, name: str, cfg: dict[str, Any], device: int | str) -> None:
        """Loads the processor and model.

        Args:
            name: Model key from the config.
            cfg: Model config section.
            device: CUDA index or ``"cpu"``.
        """
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self.name = name
        self.tile = int(cfg["tile"])
        self.batch = int(cfg["batch"])
        # decoys: look-alike classes added to the prompt so GDINO assigns e.g. a traffic cone
        # to "traffic cone" instead of "car". keep_decoys=False drops decoy-labelled boxes;
        # True returns them with their label (used by the zoom check as evidence *against* a vehicle)
        self.targets = [t.strip() for t in str(cfg["prompt"]).split(".") if t.strip()]
        self.prompt = " ".join(f"{t}." for t in self.targets + list(cfg.get("decoys", [])))
        self.keep_decoys = bool(cfg.get("keep_decoys", False))
        self.conf = float(cfg["conf"])
        self.text_threshold = float(cfg["text_threshold"])
        self.device = _torch_device(device)
        # bfloat16 autocast ~halves the time on tensor-core GPUs; float32 = exact reference
        self.dtype = getattr(torch, str(cfg.get("dtype", "float32")))
        self.autocast = self.dtype != torch.float32 and self.device.startswith("cuda")
        self.processor = AutoProcessor.from_pretrained(cfg["weights"])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg["weights"]).to(self.device).eval()

    def predict(self, images: list[np.ndarray]) -> list[Detections]:
        """Runs inference; the processor resizes the shorter side to 800 px.

        Args:
            images: BGR images of the same size (tiles) or a single full frame.

        Returns:
            One ``Detections`` per image.
        """
        rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        inputs = self.processor(images=rgb, text=[self.prompt] * len(rgb), return_tensors="pt").to(self.device)
        with self._torch.inference_mode(), self._torch.autocast("cuda", dtype=self.dtype, enabled=self.autocast):
            outputs = self.model(**inputs)
        outputs.logits, outputs.pred_boxes = outputs.logits.float(), outputs.pred_boxes.float()  # post-process in fp32
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.conf, text_threshold=self.text_threshold,
            target_sizes=[img.shape[:2] for img in images],
        )
        out = []
        for res in results:
            # "labels" is deprecated (will become integer ids); read it only on old transformers versions
            boxes = res["boxes"].float().cpu().numpy().astype(np.float32).reshape(-1, 4)
            scores = res["scores"].float().cpu().numpy().astype(np.float32)
            labels = [str(label) for label in (res["text_labels"] if "text_labels" in res else res["labels"])]
            # batch_decode([]) can return [""] for a tile without boxes: align labels to boxes
            labels = (labels + [""] * len(boxes))[:len(boxes)]
            # keep target classes; an empty label (no prompt token above text_threshold) is unknown -> keep
            keep = np.array([self.keep_decoys or is_target(label, self.targets) for label in labels], dtype=bool)
            out.append(Detections(boxes=boxes[keep], scores=scores[keep],
                                  labels=[label for label, k in zip(labels, keep) if k]))
        return out


def build_detector(name: str, cfg: dict[str, Any], device: int | str) -> Detector:
    """Creates a detector from its config section.

    Args:
        name: Model key from the config.
        cfg: Model config section with a ``type`` field.
        device: CUDA index or ``"cpu"``.

    Returns:
        Ready-to-use detector.

    Raises:
        ValueError: On an unknown model type.
    """
    if cfg["type"] in {"yolo", "rtdetr"}:
        return UltralyticsDetector(name, cfg, device)
    if cfg["type"] == "grounding_dino":
        return GroundingDinoDetector(name, cfg, device)
    raise ValueError(f"{name}: unknown model type '{cfg['type']}'")
