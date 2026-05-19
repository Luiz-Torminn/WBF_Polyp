"""Shared adapter contract and prediction container.

The contract intentionally keeps box coordinates in **pixel** space on the
original (unresized) image. Normalization to [0, 1] for Weighted Box Fusion is
the responsibility of :mod:`ensemble.fusion`, not the adapters — keeping the
shared format in pixel space makes the COCO results JSON, the visualizations,
and the evaluator all consume the same arrays without rescaling tricks.
"""

from __future__ import annotations

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
