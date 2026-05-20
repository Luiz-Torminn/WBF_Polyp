"""Weighted Box Fusion wrapper.

The :func:`weighted_boxes_fusion` function from ``ensemble-boxes`` expects
boxes normalized to [0, 1]. Adapters in this project keep boxes in pixel
coordinates on the original image, so fusion normalizes on the way in and
denormalizes on the way out. Boxes are also clipped to the unit square before
fusion to defend against off-by-one drift from upstream postprocessors.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from ensemble_boxes import weighted_boxes_fusion

from ensemble.adapters.base import Prediction


def _normalize_xyxy(xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    if xyxy.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    scale = np.array([width, height, width, height], dtype=np.float32)
    normalized = xyxy.astype(np.float32) / scale
    return np.clip(normalized, 0.0, 1.0)


def _denormalize_xyxy(xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    if xyxy.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    scale = np.array([width, height, width, height], dtype=np.float32)
    return (xyxy.astype(np.float32) * scale).astype(np.float32)


def fuse_image(
    *,
    image_id: int,
    width: int,
    height: int,
    predictions_by_model: Mapping[str, Prediction],
    weights: tuple[float, ...] | None,
    iou_thr: float,
    skip_box_thr: float,
) -> Prediction:
    """Fuse one image's per-model predictions into a single :class:`Prediction`.

    ``predictions_by_model`` is keyed by adapter name; the iteration order
    defines the per-model order seen by WBF, which is what ``weights``
    indexes into. Models that produced zero boxes contribute empty lists so
    the WBF weight assignment stays positional and unambiguous.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Non-positive image size for image_id={image_id}: ({width}, {height})")

    boxes_list: list[list[list[float]]] = []
    scores_list: list[list[float]] = []
    labels_list: list[list[int]] = []

    for prediction in predictions_by_model.values():
        normalized = _normalize_xyxy(prediction.xyxy, width, height)
        boxes_list.append(normalized.tolist() if normalized.size else [])
        scores_list.append(prediction.scores.astype(np.float32).tolist() if prediction.scores.size else [])
        labels_list.append(prediction.class_ids.astype(np.int64).tolist() if prediction.class_ids.size else [])

    if all(len(b) == 0 for b in boxes_list):
        return Prediction.empty(image_id)

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=list(weights) if weights is not None else None,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )

    fused_boxes = np.asarray(fused_boxes, dtype=np.float32)
    if fused_boxes.size == 0:
        return Prediction.empty(image_id)

    return Prediction(
        image_id=image_id,
        xyxy=_denormalize_xyxy(fused_boxes, width, height),
        scores=np.asarray(fused_scores, dtype=np.float32),
        class_ids=np.asarray(fused_labels, dtype=np.int64),
    )
