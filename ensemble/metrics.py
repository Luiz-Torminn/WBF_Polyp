"""Unified evaluator producing the four numbers reported in the CSV.

For each model (and the ensemble) the pipeline calls :func:`evaluate`. The
function builds per-image :class:`supervision.Detections` from the shared
:class:`Prediction` objects, aligns them with the in-memory ground truth
(:attr:`CocoBundle.targets`), and computes Precision, Recall, mAP@0.50 and
mAP@[0.50:0.95] with the ``supervision`` metrics (replacing the previous
``pycocotools`` evaluator, kept for reference in the gitignored
``metrics_old.py``).

All four numbers are computed at a single operating point: the predictions are
filtered to ``score >= score_threshold`` and every metric is derived from that
same filtered set. Standalone models pass their config-independent per-model
``default_conf`` so each model's baseline stays fixed regardless of ensemble
tuning; the ensemble passes ``0.0`` because its predictions are already filtered
inside Weighted Boxes Fusion at ``wbf_skip_box_thr`` — so the ensemble row
reflects the configured fusion threshold.

Precision/Recall use Supervision's defaults (``AveragingMethod.WEIGHTED``) and
the scalar is taken at IoU=0.50 (``precision_at_50`` / ``recall_at_50``). mAP is
per-class (``class_agnostic=False``). Supervision returns ``-1`` for mAP when no
ground truth is present for a class/overall; that sentinel is clamped to ``0.0``.

This module is the SINGLE source of truth for the standalone rows in
``summary.csv``. The upstream native evaluators (RFDETR ``supervision`` mAP,
Ultralytics ``model.val``, DEIMv2 ``CocoEvaluator``) still run from each upstream
directory for cross-checks but do not feed the CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision, Precision, Recall

from ensemble.adapters.base import Prediction
from ensemble.data import CocoBundle


@dataclass(frozen=True)
class EvalResult:
    precision: float
    recall: float
    map50: float
    map50_95: float


def _empty_detections() -> sv.Detections:
    """An empty prediction set that still carries a (0,) confidence array.

    ``sv.Detections.empty()`` leaves ``confidence=None``, which the Supervision
    metrics reject for predictions (they sort by confidence). Targets, by
    contrast, need no confidence, so ``bundle.targets`` may use the stock empty.
    """
    return sv.Detections(
        xyxy=np.zeros((0, 4), dtype=np.float32),
        confidence=np.zeros((0,), dtype=np.float32),
        class_id=np.zeros((0,), dtype=np.int64),
    )


def _prediction_to_detections(
    prediction: Prediction | None, score_threshold: float
) -> sv.Detections:
    """Convert a :class:`Prediction` to ``sv.Detections``, filtered by score.

    ``prediction.class_ids`` are already in the same 0-indexed ``class_idx``
    space as ``bundle.targets`` (the previous pycocotools path relied on the same
    assumption via ``class_idx_to_cat_id``), so no remapping is needed.
    """
    if prediction is None or len(prediction) == 0:
        return _empty_detections()

    scores = np.asarray(prediction.scores, dtype=np.float32)
    keep = scores >= score_threshold
    if not keep.any():
        return _empty_detections()

    return sv.Detections(
        xyxy=np.asarray(prediction.xyxy, dtype=np.float32)[keep],
        confidence=scores[keep],
        class_id=np.asarray(prediction.class_ids, dtype=np.int64)[keep],
    )


def _nonneg(value: float) -> float:
    """Clamp Supervision's ``-1`` (no-ground-truth sentinel) to ``0.0``."""
    return max(0.0, float(value))


def evaluate(
    predictions: dict[int, Prediction],
    bundle: CocoBundle,
    score_threshold: float = 0.0,
) -> EvalResult:
    """Compute Precision / Recall / mAP50 / mAP50-95 at one operating point.

    Predictions are filtered to ``score >= score_threshold`` and all four
    metrics are computed from that filtered set against ``bundle.targets``.
    """
    image_ids = sorted(bundle.targets.keys())
    if not image_ids:
        return EvalResult(precision=0.0, recall=0.0, map50=0.0, map50_95=0.0)

    targets_list = [bundle.targets[image_id] for image_id in image_ids]
    predictions_list = [
        _prediction_to_detections(predictions.get(image_id), score_threshold)
        for image_id in image_ids
    ]

    map_result = MeanAveragePrecision(class_agnostic=False).update(predictions_list, targets_list).compute()
    precision_result = Precision().update(predictions_list, targets_list).compute()
    recall_result = Recall().update(predictions_list, targets_list).compute()

    return EvalResult(
        precision=_nonneg(precision_result.precision_at_50),
        recall=_nonneg(recall_result.recall_at_50),
        map50=_nonneg(map_result.map50),
        map50_95=_nonneg(map_result.map50_95),
    )


def predictions_to_coco_results(
    predictions: dict[int, Prediction],
    class_idx_to_cat_id: dict[int, int],
) -> list[dict]:
    """Convert per-image predictions to a COCO results list."""
    results: list[dict] = []
    for image_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue
        for xyxy, score, class_id in zip(
            prediction.xyxy, prediction.scores, prediction.class_ids
        ):
            cat_id = class_idx_to_cat_id.get(int(class_id))
            if cat_id is None:
                continue
            x1, y1, x2, y2 = (float(value) for value in xyxy)
            results.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(cat_id),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    return results


def write_coco_results_json(coco_results: list[dict], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(coco_results, handle)
