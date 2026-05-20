"""Unified evaluator producing the four numbers reported in the CSV.

For each model (and the ensemble) the pipeline calls :func:`evaluate`. The
function builds an in-memory COCO results list from the shared
:class:`Prediction` objects, runs ``pycocotools`` COCOeval to get the
canonical mAP@0.50 and mAP@[0.50:0.95], and then derives a single
``(precision, recall)`` operating point at the score threshold that maximizes
mean F1 across all images at IoU=0.50. The maximum-F1 selection mirrors the
convention used by Ultralytics' validator, so the YOLO column in our CSV
remains directly comparable to ``model.val(...)`` reports without paying the
cost of running both evaluators.

This is the SINGLE source of truth for the standalone rows in
``summary.csv``. The upstream native evaluators (RFDETR ``supervision`` mAP,
Ultralytics ``model.val``, DEIMv2 ``CocoEvaluator``) still run from each
upstream directory for cross-checks but do not feed the CSV.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ensemble.adapters.base import Prediction
from ensemble.data import CocoBundle


@dataclass(frozen=True)
class EvalResult:
    precision: float
    recall: float
    map50: float
    map50_95: float
    # ``recall`` above is the best-F1 operating point on COCO's precision[T,R,K,A,M]
    # matrix. Because R is the fixed 101-point recall grid
    # (``coco_eval.params.recThrs`` = [0.00, 0.01, ..., 1.00]), ``recall`` is by
    # construction always a multiple of 0.01 — that is the source of the round
    # numbers in the CSV. The mAR fields below are NOT quantized to that grid:
    # mAR@0.50 is the mean (over classes) of recall at IoU=0.50; mAR@[0.50:0.95]
    # is ``coco_eval.stats[8]``, i.e. the standard COCO mean Average Recall at
    # maxDets=100, area=all.
    mar50: float
    mar50_95: float


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


def _build_coco_eval(
    bundle: CocoBundle, coco_results: list[dict]
) -> COCOeval | None:
    if not coco_results:
        return None

    gt = COCO(str(bundle.annotations_path))
    # `loadRes` accepts either a JSON path or a list of result dicts.
    dt = gt.loadRes(coco_results)
    coco_eval = COCOeval(gt, dt, iouType="bbox")
    coco_eval.params.imgIds = sorted(gt.getImgIds())
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
    return coco_eval


def _precision_recall_at_best_f1(coco_eval: COCOeval) -> tuple[float, float]:
    """Pick the score threshold that maximizes mean F1 at IoU=0.50.

    ``coco_eval.eval['precision']`` is a 5-D tensor with shape
    ``(T, R, K, A, M)`` where T=IoU thresholds (10), R=recall thresholds (101),
    K=classes, A=area ranges (4), M=max-detections (3). At IoU=0.50 we look at
    ``T=0``; we collapse over recall to a P/R curve and find the F1-maximizing
    point. Area range index 0 = 'all', max-dets index -1 = the largest cap (100
    in default COCO params), which is what mAP_50 also uses.
    """
    precision = coco_eval.eval.get("precision")
    recall_thresholds = coco_eval.params.recThrs
    if precision is None or precision.size == 0:
        return 0.0, 0.0

    iou_idx = 0  # IoU = 0.50
    area_idx = 0  # area = 'all'
    maxdet_idx = precision.shape[-1] - 1  # largest max-detections

    # Average precision across classes (handles single-class case trivially).
    precision_curve = precision[iou_idx, :, :, area_idx, maxdet_idx]
    valid_mask = precision_curve > -1
    if not valid_mask.any():
        return 0.0, 0.0

    # COCO sets unreachable recall bins to -1; mask them out before averaging.
    precision_curve = np.where(valid_mask, precision_curve, 0.0)
    mean_precision = precision_curve.mean(axis=1)  # shape (R,)
    recall_values = np.asarray(recall_thresholds, dtype=np.float64)

    denominator = mean_precision + recall_values
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denominator > 0.0, 2.0 * mean_precision * recall_values / denominator, 0.0)

    best_idx = int(np.argmax(f1))
    return float(mean_precision[best_idx]), float(recall_values[best_idx])


def _mean_average_recall_50(coco_eval: COCOeval) -> float:
    """mAR at IoU=0.50, maxDets=largest, area=all, averaged over classes.

    ``coco_eval.eval['recall']`` has shape ``(T, K, A, M)`` (no R dimension —
    unlike precision, recall is per IoU threshold). We slice IoU index 0,
    area index 0, the largest max-detections, and average over classes.
    Unreachable bins are marked with -1 and masked out before averaging.
    """
    recall_array = coco_eval.eval.get("recall")
    if recall_array is None or recall_array.size == 0:
        return 0.0
    iou_idx = 0
    area_idx = 0
    maxdet_idx = recall_array.shape[-1] - 1
    recall_slice = recall_array[iou_idx, :, area_idx, maxdet_idx]
    valid = recall_slice > -1
    if not valid.any():
        return 0.0
    return float(recall_slice[valid].mean())


def evaluate(
    predictions: dict[int, Prediction],
    bundle: CocoBundle,
) -> EvalResult:
    """Compute Precisão / Recall / mAP50 / mAP50-95 / mAR50 / mAR50-95 for a single model."""
    coco_results = predictions_to_coco_results(predictions, bundle.class_idx_to_cat_id)

    empty = EvalResult(
        precision=0.0,
        recall=0.0,
        map50=0.0,
        map50_95=0.0,
        mar50=0.0,
        mar50_95=0.0,
    )

    if not coco_results:
        return empty

    coco_eval = _build_coco_eval(bundle, coco_results)
    if coco_eval is None:
        return empty

    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()

    stats = coco_eval.stats
    map50_95 = float(stats[0])
    map50 = float(stats[1])
    # stats[8] is mAR @ maxDets=100, area=all, averaged over IoU 0.50:0.95.
    mar50_95 = float(stats[8])
    mar50 = _mean_average_recall_50(coco_eval)
    precision, recall = _precision_recall_at_best_f1(coco_eval)

    return EvalResult(
        precision=precision,
        recall=recall,
        map50=map50,
        map50_95=map50_95,
        mar50=mar50,
        mar50_95=mar50_95,
    )


def write_coco_results_json(coco_results: list[dict], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(coco_results, handle)
