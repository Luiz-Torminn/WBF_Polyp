"""Unified evaluator producing the four numbers reported in the CSV.

For each model (and the ensemble) the pipeline calls :func:`evaluate`. The
function builds per-image :class:`supervision.Detections` from the shared
:class:`Prediction` objects, aligns them with the in-memory ground truth
(:attr:`CocoBundle.targets`), and computes Precision, Recall, mAP@0.50 and
mAP@[0.50:0.95] with the ``supervision`` metrics — replacing the previous
``pycocotools`` evaluator (kept for reference in the gitignored
``metrics_old.py``).

The metric *calculation* is identical for every row: supervision defaults
(``MeanAveragePrecision(class_agnostic=False)`` for mAP; ``Precision`` / ``Recall``
with ``averaging_method=WEIGHTED``, read at IoU=0.50). Only the *predictions*
differ — standalone rows come from each model's validation-default inference pass,
the ENSEMBLE row from the config/WBF-fused predictions.

Precision/Recall are read at a fixed **confidence operating point**
(:data:`PR_CONFIDENCE_THRESHOLD`, 0.5): only detections with ``confidence >= 0.5``
feed the P/R metrics. This replaces the previous "all detections, no operating
point" behavior, which drove standalone precision toward zero for the DETR models
(RFDETR/DEIMv2 emit ~300 low-confidence queries per image with no NMS). mAP is
computed from the FULL, unfiltered prediction set — it needs the low-confidence
tail for the PR curve — so only P/R are filtered.

Notes:
  * Supervision's Precision/Recall have NO confidence-threshold parameter and no
    default (confidence is used only to sort detections); the operating point is
    therefore enforced here by filtering the ``sv.Detections`` before ``.update()``.
  * The threshold uses ``>=`` (a box exactly at 0.5 is kept).
  * Supervision returns ``-1`` for mAP when a class/overall has no ground truth;
    that sentinel is clamped to ``0.0``.
  * Prediction ``sv.Detections`` must carry a ``confidence`` array even when empty
    (the metrics sort by confidence), so empties are built explicitly rather than
    via ``sv.Detections.empty()``. Confidence filtering preserves this: a mask that
    removes every box still yields a ``(0,)`` confidence array.

This module is the SINGLE source of truth for the standalone rows in
``summary.csv``. The upstream native evaluators (RFDETR ``supervision`` mAP,
Ultralytics ``model.val``, DEIMv2 ``CocoEvaluator``) still run from each upstream
directory for cross-checks but do not feed the CSV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision, Precision, Recall

from ensemble.adapters.base import Prediction
from ensemble.data import CocoBundle

# Fixed confidence operating point for Precision/Recall. Detections below this
# score are excluded from P/R (but not from mAP). Supervision exposes no
# threshold of its own, so this is applied here by filtering the detections.
PR_CONFIDENCE_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class EvalResult:
    precision: float
    recall: float
    map50: float
    map50_95: float


def _empty_detections() -> sv.Detections:
    """An empty prediction set that still carries a ``(0,)`` confidence array.

    ``sv.Detections.empty()`` may leave ``confidence`` unset, which the
    Supervision metrics reject for predictions (they sort by confidence). Targets
    need no confidence, so ``bundle.targets`` may use the stock empty.
    """
    return sv.Detections(
        xyxy=np.zeros((0, 4), dtype=np.float32),
        confidence=np.zeros((0,), dtype=np.float32),
        class_id=np.zeros((0,), dtype=np.int64),
    )


def _prediction_to_detections(prediction: Prediction | None) -> sv.Detections:
    """Convert a :class:`Prediction` to ``sv.Detections`` (no score filtering).

    ``prediction.class_ids`` are already in the same 0-indexed ``class_idx`` space
    as ``bundle.targets.class_id`` (the previous pycocotools path relied on the
    same assumption via ``class_idx_to_cat_id``), so no remapping is needed.
    """
    if prediction is None or len(prediction) == 0:
        return _empty_detections()

    return sv.Detections(
        xyxy=np.asarray(prediction.xyxy, dtype=np.float32),
        confidence=np.asarray(prediction.scores, dtype=np.float32),
        class_id=np.asarray(prediction.class_ids, dtype=np.int64),
    )


def _nonneg(value: float) -> float:
    """Clamp Supervision's ``-1`` (no-ground-truth sentinel) to ``0.0``."""
    return max(0.0, float(value))


def _filter_by_confidence(
    detections: sv.Detections, threshold: float
) -> sv.Detections:
    """Keep only detections scoring at or above ``threshold`` (``>=``).

    A mask that removes every box still returns a ``Detections`` with a ``(0,)``
    ``confidence`` array, which the Supervision metrics require.
    """
    if len(detections) == 0:
        return detections
    return detections[detections.confidence >= threshold]


def evaluate(
    predictions: dict[int, Prediction],
    bundle: CocoBundle,
    pr_confidence_threshold: float = PR_CONFIDENCE_THRESHOLD,
) -> EvalResult:
    """Compute Precision / Recall / mAP50 / mAP50-95 with supervision.

    mAP is computed from ``predictions`` as given (unfiltered — the full
    low-confidence tail is needed for the PR curve). Precision and Recall are
    computed only over detections with ``confidence >= pr_confidence_threshold``
    (the fixed operating point). All metrics are aligned per image over the
    sorted target keys.
    """
    image_ids = sorted(bundle.targets.keys())
    if not image_ids:
        return EvalResult(precision=0.0, recall=0.0, map50=0.0, map50_95=0.0)

    targets_list = [bundle.targets[image_id] for image_id in image_ids]
    predictions_list = [
        _prediction_to_detections(predictions.get(image_id)) for image_id in image_ids
    ]
    pr_predictions_list = [
        _filter_by_confidence(detections, pr_confidence_threshold)
        for detections in predictions_list
    ]

    map_result = (
        MeanAveragePrecision(class_agnostic=False)
        .update(predictions_list, targets_list)
        .compute()
    )
    precision_result = Precision().update(pr_predictions_list, targets_list).compute()
    recall_result = Recall().update(pr_predictions_list, targets_list).compute()

    return EvalResult(
        precision=_nonneg(precision_result.precision_at_50),
        recall=_nonneg(recall_result.recall_at_50),
        map50=_nonneg(map_result.map50),
        map50_95=_nonneg(map_result.map50_95),
    )
