"""Camera geometry, tiling and box utilities (pure numpy, no model dependencies)."""

from __future__ import annotations

import math

import numpy as np


def focal_px(width: int, height: int, hfov_deg: float) -> float:
    """Focal length in pixels, assuming the HFOV applies to the frame's long side.

    The eval clip is portrait (2160x3840): the drone rotated the sensor, so the
    82 deg field of view spans the 3840 px side, not the 2160 px one.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        hfov_deg: Field of view across the sensor's long side, degrees.

    Returns:
        Focal length in pixels.
    """
    long_side = max(width, height)
    return long_side / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def box_px_at(distance_m: float, f_px: float, ref_length_m: float) -> float:
    """Expected long side of a vehicle box at a given slant range (pinhole model).

    Args:
        distance_m: Slant range in metres.
        f_px: Focal length in pixels.
        ref_length_m: Reference vehicle length in metres.

    Returns:
        Box long side in pixels.
    """
    return f_px * ref_length_m / distance_m


def tile_origins(length: int, tile: int, overlap: float) -> list[int]:
    """Start offsets of tiles along one axis; the last tile is aligned to the border.

    Example: length=3840, tile=640, overlap=0.2 -> stride 512 ->
    [0, 512, ..., 3072, 3200] (8 tiles). Any object up to ``tile * overlap`` px
    lies fully inside at least one tile.

    Args:
        length: Image size along the axis.
        tile: Tile size in pixels.
        overlap: Fraction of the tile shared with the neighbour.

    Returns:
        Sorted tile start offsets.
    """
    if length <= tile:
        return [0]
    stride = max(1, round(tile * (1.0 - overlap)))
    origins = list(range(0, length - tile, stride))
    origins.append(length - tile)
    return origins


def tile_windows(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """All tile windows of an image.

    Args:
        width: Image width.
        height: Image height.
        tile: Square tile size.
        overlap: Fraction of overlap between neighbouring tiles.

    Returns:
        Windows as ``(x0, y0, x1, y1)`` in image pixels, row-major order.
    """
    return [
        (x0, y0, min(x0 + tile, width), min(y0 + tile, height))
        for y0 in tile_origins(height, tile, overlap)
        for x0 in tile_origins(width, tile, overlap)
    ]


def touches_inner_edge(
    boxes: np.ndarray, window: tuple[int, int, int, int], image_size: tuple[int, int], margin: float
) -> np.ndarray:
    """Flags boxes cut by a tile edge that is inside the image (not the image border).

    Such boxes are usually fragments of a vehicle that is fully visible in the
    neighbouring tile (overlap) or in the full-frame pass, so they are dropped.

    Args:
        boxes: ``(N, 4)`` xyxy boxes in tile coordinates.
        window: Tile window ``(x0, y0, x1, y1)`` in image coordinates.
        image_size: ``(width, height)`` of the full image.
        margin: Distance in pixels to an edge that counts as touching.

    Returns:
        Boolean mask of shape ``(N,)``.
    """
    x0, y0, x1, y1 = window
    width, height = image_size
    tw, th = x1 - x0, y1 - y0
    cut = np.zeros(len(boxes), dtype=bool)
    if x0 > 0:
        cut |= boxes[:, 0] <= margin
    if y0 > 0:
        cut |= boxes[:, 1] <= margin
    if x1 < width:
        cut |= boxes[:, 2] >= tw - margin
    if y1 < height:
        cut |= boxes[:, 3] >= th - margin
    return cut


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one box and many boxes.

    Args:
        box: ``(4,)`` xyxy box.
        boxes: ``(N, 4)`` xyxy boxes.

    Returns:
        ``(N,)`` IoU values.
    """
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def nwd_one_to_many(box: np.ndarray, boxes: np.ndarray, c: float) -> np.ndarray:
    """Normalized Wasserstein Distance similarity (Wang et al., 2021) for tiny boxes.

    Each box is a 2D Gaussian (centre, half-sizes); unlike IoU the similarity
    decays smoothly with centre offset, so two slightly shifted 12 px boxes still
    match. It is in absolute pixels, hence used only for far (small) objects.

    Args:
        box: ``(4,)`` xyxy box.
        boxes: ``(N, 4)`` xyxy boxes.
        c: Normalising constant in pixels (typical object size).

    Returns:
        ``(N,)`` similarities in ``(0, 1]``.
    """
    def gauss(b: np.ndarray) -> np.ndarray:
        return np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2,
                         (b[..., 2] - b[..., 0]) / 2, (b[..., 3] - b[..., 1]) / 2], axis=-1)

    w2 = np.sqrt(((gauss(boxes) - gauss(box)) ** 2).sum(axis=-1))
    return np.exp(-w2 / c)


def contained_fraction(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Share of each box in ``boxes`` that lies inside ``box``.

    Args:
        box: ``(4,)`` xyxy container candidate.
        boxes: ``(N, 4)`` xyxy boxes.

    Returns:
        ``(N,)`` intersection / area of each box in ``boxes``.
    """
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(areas, 1e-9)


def distance_m(boxes: np.ndarray, f_px: float, ref_length_m: float) -> np.ndarray:
    """Slant range of each box from its long side (pinhole model, see :func:`box_px_at`).

    Args:
        boxes: ``(N, 4)`` xyxy boxes.
        f_px: Focal length in pixels.
        ref_length_m: Reference vehicle length in metres.

    Returns:
        ``(N,)`` distances in metres.
    """
    long_side = np.maximum(boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1])
    return f_px * ref_length_m / np.maximum(long_side, 1e-6)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy class-agnostic non-maximum suppression.

    Args:
        boxes: ``(N, 4)`` xyxy boxes.
        scores: ``(N,)`` confidences.
        iou_thr: Boxes overlapping a kept box above this IoU are removed.

    Returns:
        Indices of kept boxes, highest score first.
    """
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        order = rest[iou_one_to_many(boxes[i], boxes[rest]) < iou_thr]
    return np.asarray(keep, dtype=int)
