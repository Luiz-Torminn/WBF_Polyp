"""Ultralytics YOLO adapter.

Uses ``YOLO.predict(...)`` rather than ``YOLO.val(...)`` because the ensemble
pipeline must consume predictions per-image keyed by COCO ``image_id``. The
upstream ``YOLO_model/main.py`` keeps its native ``model.val`` flow intact for
the standalone YOLO eval — this adapter only powers the unified pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ensemble.adapters.base import Prediction
from ensemble.data import ImageRecord


_YOLO_PATH_INJECTED = False


def _ensure_ultralytics_importable(yolo_dir: Path) -> None:
    """Front-load ``YOLO_model/`` on sys.path so ``import ultralytics`` picks up
    the in-tree package even when the editable install pointer is stale.
    """
    global _YOLO_PATH_INJECTED
    if _YOLO_PATH_INJECTED:
        return
    yolo_dir = Path(yolo_dir).resolve()
    if not (yolo_dir / "ultralytics").is_dir():
        raise FileNotFoundError(
            f"Expected an 'ultralytics/' package directory under: {yolo_dir}"
        )
    yolo_str = str(yolo_dir)
    if yolo_str not in sys.path:
        sys.path.insert(0, yolo_str)
    _YOLO_PATH_INJECTED = True


class YOLOAdapter:
    name = "yolo"
    display_name = "YOLOv12 nano"

    def __init__(
        self,
        weights_path: Path,
        predict_threshold: float,
        iou_threshold: float = 0.7,
        imgsz: int = 640,
        yolo_dir: Path | None = None,
    ):
        self._weights_path = Path(weights_path)
        self._predict_threshold = float(predict_threshold)
        self._iou_threshold = float(iou_threshold)
        self._imgsz = int(imgsz)
        self._yolo_dir = Path(yolo_dir) if yolo_dir is not None else self._weights_path.parent
        self._model = None
        self._device: str = "cpu"

    def load(self, device: str) -> None:
        if self._model is not None:
            return
        if not self._weights_path.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {self._weights_path}")
        _ensure_ultralytics_importable(self._yolo_dir)
        from ultralytics import YOLO

        self._device = device
        self._model = YOLO(str(self._weights_path))
        # Warm-up to materialize the model on the target device.
        self._model.to(device)

    def infer_batch(self, batch: list[ImageRecord]) -> list[Prediction]:
        if self._model is None:
            raise RuntimeError("YOLOAdapter.load() must be called before infer_batch")

        paths = [str(rec.path) for rec in batch]
        results = self._model.predict(
            paths,
            conf=self._predict_threshold,
            iou=self._iou_threshold,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )

        predictions: list[Prediction] = []
        for record, result in zip(batch, results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                predictions.append(Prediction.empty(record.image_id))
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32, copy=False)
            scores = boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)
            class_ids = boxes.cls.detach().cpu().numpy().astype(np.int64, copy=False)
            predictions.append(
                Prediction(
                    image_id=record.image_id,
                    xyxy=xyxy,
                    scores=scores,
                    class_ids=class_ids,
                )
            )
        return predictions

    def unload(self) -> None:
        if self._model is None:
            return
        try:
            import torch

            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            self._model = None
