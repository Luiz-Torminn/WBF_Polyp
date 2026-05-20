"""Shared adapter contract and prediction container.

The contract intentionally keeps box coordinates in **pixel** space on the
original (unresized) image. Normalization to [0, 1] for Weighted Box Fusion is
the responsibility of :mod:`ensemble.fusion`, not the adapters — keeping the
shared format in pixel space makes the COCO results JSON, the visualizations,
and the evaluator all consume the same arrays without rescaling tricks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ensemble.data import ImageRecord


@dataclass(frozen=True)
class Prediction:
    """Detection result for a single image in a shared, model-agnostic form."""

    image_id: int
    xyxy: np.ndarray         # (N, 4) float32 in pixel coords on the ORIGINAL image
    scores: np.ndarray       # (N,)   float32
    class_ids: np.ndarray    # (N,)   int64, 0-indexed model class id

    @classmethod
    def empty(cls, image_id: int) -> "Prediction":
        return cls(
            image_id=image_id,
            xyxy=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            class_ids=np.zeros((0,), dtype=np.int64),
        )

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


class Adapter(Protocol):
    """Lifecycle and inference contract every model adapter must satisfy."""

    name: str
    display_name: str

    def load(self, device: str) -> None: ...

    def infer_batch(self, batch: list[ImageRecord]) -> list[Prediction]: ...

    def unload(self) -> None: ...


def log_batch_predictions(
    adapter_name: str,
    predictions: list[Prediction],
    *,
    sample_first: int = 5,
    include_samples: bool = False,
) -> None:
    """Emit a one-line per-batch summary plus optional first-N detection dump.

    Activated only when the ``ensemble.adapters.<name>`` logger is at DEBUG.
    The summary stays small (image count, detection count, score min/mean/max)
    so it is safe to call on every batch even on a 1600+ image test split.
    The detail dump is opt-in (``include_samples=True``) and intended to be
    set only on the first batch of each adapter run — see how each adapter
    flips its ``_first_batch_logged`` flag.
    """
    logger = logging.getLogger(f"ensemble.adapters.{adapter_name}")
    if not logger.isEnabledFor(logging.DEBUG):
        return

    num_images = len(predictions)
    per_image_counts = [len(pred) for pred in predictions]
    total_detections = int(sum(per_image_counts))

    all_scores = np.concatenate(
        [pred.scores for pred in predictions if len(pred) > 0]
    ) if total_detections > 0 else np.zeros((0,), dtype=np.float32)

    if all_scores.size:
        score_summary = (
            f"min={float(all_scores.min()):.4f} "
            f"mean={float(all_scores.mean()):.4f} "
            f"max={float(all_scores.max()):.4f}"
        )
    else:
        score_summary = "no detections"

    logger.debug(
        "%s batch: images=%d, detections=%d (per-image min/max=%d/%d), scores: %s",
        adapter_name,
        num_images,
        total_detections,
        min(per_image_counts) if per_image_counts else 0,
        max(per_image_counts) if per_image_counts else 0,
        score_summary,
    )

    if not include_samples or total_detections == 0:
        return

    shown = 0
    for prediction in predictions:
        if shown >= sample_first:
            break
        if len(prediction) == 0:
            continue
        # Take the highest-scoring detections per image so the sample isn't
        # dominated by the low-confidence tail at predict_threshold=0.001.
        order = np.argsort(-prediction.scores)
        for idx in order:
            if shown >= sample_first:
                break
            xyxy = prediction.xyxy[idx]
            logger.debug(
                "  sample #%d: image_id=%d class=%d score=%.4f xyxy=[%.1f, %.1f, %.1f, %.1f]",
                shown + 1,
                int(prediction.image_id),
                int(prediction.class_ids[idx]),
                float(prediction.scores[idx]),
                float(xyxy[0]),
                float(xyxy[1]),
                float(xyxy[2]),
                float(xyxy[3]),
            )
            shown += 1
