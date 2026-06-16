"""COCO dataset access and ground-truth target construction.

Mirrors the loader pattern in ``RFDETR/main.py`` so the unified pipeline sees
exactly the same image manifest and category indexing each upstream eval uses.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import supervision as sv

logger = logging.getLogger("ensemble.data")


@dataclass(frozen=True)
class ImageRecord:
    """Per-image manifest entry shared across adapters and the evaluator."""

    image_id: int
    file_name: str
    path: Path
    width: int
    height: int


@dataclass
class CocoBundle:
    """Container for the COCO ground truth used by both inference and eval."""

    annotations_path: Path
    raw: dict
    cat_id_to_class_idx: dict[int, int]
    class_idx_to_cat_id: dict[int, int]
    image_records: list[ImageRecord]
    targets: dict[int, sv.Detections]

    @property
    def num_classes(self) -> int:
        return len(self.cat_id_to_class_idx)

    @property
    def category_names(self) -> list[str]:
        idx_to_name = {
            self.cat_id_to_class_idx[cat["id"]]: cat["name"]
            for cat in self.raw["categories"]
            if cat["id"] in self.cat_id_to_class_idx
        }
        return [idx_to_name[i] for i in range(len(idx_to_name))]


def load_coco(annotations_path: Path, images_dir: Path) -> CocoBundle:
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Missing annotations file: {annotations_path}")

    with open(annotations_path, encoding="utf-8") as handle:
        coco = json.load(handle)

    sorted_cat_ids = sorted(cat["id"] for cat in coco["categories"])
    cat_id_to_class_idx = {cat_id: idx for idx, cat_id in enumerate(sorted_cat_ids)}
    class_idx_to_cat_id = {idx: cat_id for cat_id, idx in cat_id_to_class_idx.items()}

    image_records: list[ImageRecord] = []
    for image in coco["images"]:
        image_records.append(
            ImageRecord(
                image_id=int(image["id"]),
                file_name=image["file_name"],
                path=images_dir / image["file_name"],
                width=int(image["width"]),
                height=int(image["height"]),
            )
        )

    targets = _build_targets(coco, cat_id_to_class_idx)

    if logger.isEnabledFor(logging.DEBUG):
        annotations = coco.get("annotations", [])
        per_image_gt = [len(targets[record.image_id]) for record in image_records]
        empty_images = sum(1 for count in per_image_gt if count == 0)
        logger.debug(
            "load_coco: file=%s images=%d annotations=%d categories=%d empty_images=%d",
            annotations_path,
            len(image_records),
            len(annotations),
            len(cat_id_to_class_idx),
            empty_images,
        )
        logger.debug(
            "load_coco: per-image GT count min=%d max=%d mean=%.2f",
            min(per_image_gt) if per_image_gt else 0,
            max(per_image_gt) if per_image_gt else 0,
            (sum(per_image_gt) / len(per_image_gt)) if per_image_gt else 0.0,
        )
        logger.debug(
            "load_coco: category map (cat_id -> class_idx) = %s",
            cat_id_to_class_idx,
        )
        for rec in image_records[:3]:
            logger.debug(
                "  sample image: id=%d name=%s size=%dx%d path=%s",
                rec.image_id,
                rec.file_name,
                rec.width,
                rec.height,
                rec.path,
            )
        for rec in image_records[:3]:
            target = targets[rec.image_id]
            if len(target) == 0:
                logger.debug("  sample target image_id=%d: <empty>", rec.image_id)
                continue
            xyxy_first = target.xyxy[0]
            logger.debug(
                "  sample target image_id=%d: count=%d first_xyxy=[%.1f, %.1f, %.1f, %.1f] first_class=%d",
                rec.image_id,
                len(target),
                float(xyxy_first[0]),
                float(xyxy_first[1]),
                float(xyxy_first[2]),
                float(xyxy_first[3]),
                int(target.class_id[0]),
            )

    return CocoBundle(
        annotations_path=annotations_path,
        raw=coco,
        cat_id_to_class_idx=cat_id_to_class_idx,
        class_idx_to_cat_id=class_idx_to_cat_id,
        image_records=image_records,
        targets=targets,
    )


def _build_targets(
    coco: dict, cat_id_to_class_idx: dict[int, int]
) -> dict[int, sv.Detections]:
    annotations_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    targets: dict[int, sv.Detections] = {}
    for image in coco["images"]:
        image_id = int(image["id"])
        image_anns = annotations_by_image.get(image_id, [])
        if not image_anns:
            targets[image_id] = sv.Detections.empty()
            continue

        xyxy = np.array(
            [
                [
                    ann["bbox"][0],
                    ann["bbox"][1],
                    ann["bbox"][0] + ann["bbox"][2],
                    ann["bbox"][1] + ann["bbox"][3],
                ]
                for ann in image_anns
            ],
            dtype=np.float32,
        )
        class_id = np.array(
            [cat_id_to_class_idx[ann["category_id"]] for ann in image_anns],
            dtype=int,
        )
        targets[image_id] = sv.Detections(xyxy=xyxy, class_id=class_id)

    return targets


def iter_batches(records: list[ImageRecord], batch_size: int):
    """Yield contiguous batches of ImageRecord objects."""
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]
