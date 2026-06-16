"""RFDETR Nano adapter.

Mirrors the inference pattern in ``RFDETR/main.py``: instantiate
``RFDETRNano(pretrain_weights=...)``, call ``optimize_for_inference(compile=False)``
so variable-sized last batches don't trip the compiled graph, then call
``predict(...)``. Output ``sv.Detections`` are converted into the shared
:class:`Prediction` form in pixel coords on the original image.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from ensemble.adapters.base import Prediction, log_batch_predictions
from ensemble.data import ImageRecord

logger = logging.getLogger("ensemble.adapters.rfdetr")


class RFDETRAdapter:
    name = "rfdetr"
    display_name = "RFDETR nano"

    def __init__(self, weights_path: Path, predict_threshold: float):
        self._weights_path = Path(weights_path)
        self._predict_threshold = float(predict_threshold)
        self._model = None
        self._first_batch_logged = False

    def load(self, device: str) -> None:
        if self._model is not None:
            return
        from rfdetr import RFDETRNano

        if not self._weights_path.is_file():
            raise FileNotFoundError(f"RFDETR weights not found: {self._weights_path}")

        # `device` is honored implicitly by RFDETRNano via its internal cuda
        # selection. compile=False avoids retracing on variable batch sizes.
        self._model = RFDETRNano(pretrain_weights=str(self._weights_path))
        self._model.optimize_for_inference(compile=False)
        self._first_batch_logged = False
        logger.debug(
            "loaded weights=%s device=%s threshold=%.4f",
            self._weights_path,
            device,
            self._predict_threshold,
        )

    def infer_batch(self, batch: list[ImageRecord]) -> list[Prediction]:
        if self._model is None:
            raise RuntimeError("RFDETRAdapter.load() must be called before infer_batch")

        images = [Image.open(rec.path).convert("RGB") for rec in batch]
        results = self._model.predict(
            images,
            threshold=self._predict_threshold,
            include_source_image=False,
        )
        if not isinstance(results, list):
            results = [results]

        predictions: list[Prediction] = []
        for record, detections in zip(batch, results):
            if len(detections) == 0:
                predictions.append(Prediction.empty(record.image_id))
                continue
            predictions.append(
                Prediction(
                    image_id=record.image_id,
                    xyxy=np.asarray(detections.xyxy, dtype=np.float32),
                    scores=np.asarray(detections.confidence, dtype=np.float32),
                    class_ids=np.asarray(detections.class_id, dtype=np.int64),
                )
            )

        log_batch_predictions(
            "rfdetr",
            predictions,
            include_samples=not self._first_batch_logged,
        )
        self._first_batch_logged = True
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
