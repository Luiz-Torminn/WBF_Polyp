"""Per-image visualization helpers.

The pipeline samples a handful of images that contain at least one ground
truth annotation and writes one overlay per model plus one for the fused
ensemble, so users can sanity-check the WBF output by eye. Top-K filtering
keeps overlays readable when the predict threshold is intentionally tiny.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image

from ensemble.adapters.base import Prediction
from ensemble.data import ImageRecord


_BOX_ANNOTATOR = sv.BoxAnnotator(thickness=2)
_LABEL_ANNOTATOR = sv.LabelAnnotator()


def select_visualization_records(
    image_records: list[ImageRecord],
    targets,
    count: int,
) -> list[ImageRecord]:
    """Pick up to ``count`` images that actually contain ground truth boxes."""
    selected: list[ImageRecord] = []
    for record in image_records:
        target = targets.get(record.image_id)
        if target is None or len(target) == 0:
            continue
        selected.append(record)
        if len(selected) >= count:
            break
    return selected


def _to_detections(prediction: Prediction, top_k: int) -> sv.Detections:
    if len(prediction) == 0:
        return sv.Detections.empty()
    if top_k > 0 and len(prediction) > top_k:
        order = np.argsort(-prediction.scores)[:top_k]
        xyxy = prediction.xyxy[order]
        scores = prediction.scores[order]
        class_ids = prediction.class_ids[order]
    else:
        xyxy = prediction.xyxy
        scores = prediction.scores
        class_ids = prediction.class_ids
    return sv.Detections(
        xyxy=xyxy.astype(np.float32),
        confidence=scores.astype(np.float32),
        class_id=class_ids.astype(int),
    )


def _annotate(image: np.ndarray, detections: sv.Detections, class_names: list[str]) -> np.ndarray:
    if len(detections) == 0:
        return image
    labels = [
        f"{class_names[cls] if 0 <= cls < len(class_names) else cls} {conf:.2f}"
        for cls, conf in zip(detections.class_id, detections.confidence)
    ]
    annotated = _BOX_ANNOTATOR.annotate(scene=image, detections=detections)
    annotated = _LABEL_ANNOTATOR.annotate(scene=annotated, detections=detections, labels=labels)
    return annotated


def write_overlays(
    record: ImageRecord,
    predictions: dict[str, Prediction],
    class_names: list[str],
    output_dir: Path,
    top_k: int = 10,
) -> None:
    """Write one ``<stem>_<model>.jpg`` overlay per model under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base_image = np.array(Image.open(record.path).convert("RGB"))

    stem = Path(record.file_name).stem
    for model_key, prediction in predictions.items():
        detections = _to_detections(prediction, top_k=top_k)
        annotated = _annotate(base_image.copy(), detections, class_names)
        out_path = output_dir / f"{stem}_{model_key}.jpg"
        Image.fromarray(annotated).save(out_path, quality=92)
