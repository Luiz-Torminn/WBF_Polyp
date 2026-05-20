"""Per-image visualization helpers.

The pipeline samples a handful of images that contain at least one ground
truth annotation and writes one overlay per model plus one for the fused
ensemble, so users can sanity-check the WBF output by eye. Top-K filtering
keeps overlays readable when the predict threshold is intentionally tiny.

Each model has a distinct color so that the per-model JPGs are immediately
distinguishable AND a single ``<stem>_combined.jpg`` can layer all per-model
boxes plus the surviving ensemble boxes on one image — useful for seeing
which candidates WBF kept vs. merged vs. dropped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image

from ensemble.adapters.base import Prediction
from ensemble.data import ImageRecord


# Color per source. The ensemble is drawn in a bold contrasting color so it
# pops on the combined overlay even when individual-model boxes cluster on
# top of each other.
MODEL_COLORS: dict[str, sv.Color] = {
    "rfdetr": sv.Color(r=30, g=144, b=255),    # dodger blue
    "yolo": sv.Color(r=0, g=200, b=83),        # green
    "deimv2": sv.Color(r=255, g=193, b=7),     # amber
    "ensemble": sv.Color(r=229, g=57, b=53),   # red
}

# Thicker line for the ensemble so it visibly dominates the combined overlay.
_ENSEMBLE_THICKNESS = 3
_MODEL_THICKNESS = 2


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


def _annotators_for(model_key: str) -> tuple[sv.BoxAnnotator, sv.LabelAnnotator]:
    color = MODEL_COLORS.get(model_key, sv.Color(r=200, g=200, b=200))
    thickness = _ENSEMBLE_THICKNESS if model_key == "ensemble" else _MODEL_THICKNESS
    box = sv.BoxAnnotator(color=color, thickness=thickness)
    label = sv.LabelAnnotator(color=color, text_color=sv.Color.WHITE)
    return box, label


def _annotate(
    image: np.ndarray,
    detections: sv.Detections,
    class_names: list[str],
    model_key: str,
    *,
    include_labels: bool = True,
) -> np.ndarray:
    if len(detections) == 0:
        return image
    box_annotator, label_annotator = _annotators_for(model_key)
    annotated = box_annotator.annotate(scene=image, detections=detections)
    if include_labels:
        labels = [
            f"{class_names[cls] if 0 <= cls < len(class_names) else cls} {conf:.2f}"
            for cls, conf in zip(detections.class_id, detections.confidence)
        ]
        annotated = label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )
    return annotated


def write_overlays(
    record: ImageRecord,
    predictions: dict[str, Prediction],
    class_names: list[str],
    output_dir: Path,
    top_k: int = 10,
) -> None:
    """Write one ``<stem>_<model>.jpg`` overlay per model under ``output_dir``.

    Each overlay uses the color assigned to that model in :data:`MODEL_COLORS`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_image = np.array(Image.open(record.path).convert("RGB"))

    stem = Path(record.file_name).stem
    for model_key, prediction in predictions.items():
        detections = _to_detections(prediction, top_k=top_k)
        annotated = _annotate(base_image.copy(), detections, class_names, model_key)
        out_path = output_dir / f"{stem}_{model_key}.jpg"
        Image.fromarray(annotated).save(out_path, quality=92)


def write_combined_overlay(
    record: ImageRecord,
    predictions: dict[str, Prediction],
    class_names: list[str],
    output_dir: Path,
    top_k: int = 10,
) -> None:
    """Write a single ``<stem>_combined.jpg`` showing all sources at once.

    Per-model boxes are drawn first (thin, in their own color, without labels
    to reduce clutter); the ensemble boxes are drawn on top with a thicker
    stroke and labels so the surviving fused detections are immediately
    distinguishable. If an ``ensemble`` key is missing from ``predictions``
    the overlay still renders the per-model layers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_image = np.array(Image.open(record.path).convert("RGB"))
    canvas = base_image.copy()

    # Draw per-model layers first, ensemble last so it dominates the image.
    per_model_keys = [k for k in predictions.keys() if k != "ensemble"]
    for model_key in per_model_keys:
        detections = _to_detections(predictions[model_key], top_k=top_k)
        canvas = _annotate(
            canvas, detections, class_names, model_key, include_labels=False
        )

    if "ensemble" in predictions:
        detections = _to_detections(predictions["ensemble"], top_k=top_k)
        canvas = _annotate(canvas, detections, class_names, "ensemble")

    stem = Path(record.file_name).stem
    out_path = output_dir / f"{stem}_combined.jpg"
    Image.fromarray(canvas).save(out_path, quality=92)
