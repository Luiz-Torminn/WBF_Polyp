"""Unified evaluator producing the four numbers reported in the CSV.

For each model (and the ensemble) the pipeline calls :func:`evaluate`. The
function builds an in-memory COCO results list from the shared
:class:`Prediction` objects, runs ``pycocotools`` COCOeval to get the
canonical mAP@0.50 and mAP@[0.50:0.95], and then derives a single
``(precision, recall)`` operating point at the score threshold that maximizes
F1 across all detections at IoU=0.50.

The (P, R) operating point is reconstructed from the **raw per-detection
TP/FP** stored on ``coco_eval.evalImgs`` rather than read out of the binned
``coco_eval.eval['precision']`` matrix. The matrix-based shortcut would force
the recall axis onto the fixed 101-point ``recThrs`` grid (multiples of 0.01)
which silently quantizes the reported Recall in the CSV. Sorting detections
by score and walking the cumulative TP/FP curve gives a **continuous** F1-max
operating point, matching Ultralytics' ``box.mp`` / ``box.mr`` convention.

This module is the SINGLE source of truth for the standalone rows in
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
    """Pick the score threshold that maximizes F1 at IoU=0.50 — continuous.

    Walks ``coco_eval.evalImgs`` to pull per-detection ``(score, TP@0.50,
    ignore)`` triples plus the total non-ignored GT count, then reconstructs
    the cumulative TP / FP curve in descending-score order. Precision /
    recall / F1 are evaluated at every detection step; the F1-argmax detection
    is the reported operating point. Equivalent to Ultralytics' ``box.mp`` /
    ``box.mr`` (modulo a small EMA smoothing they apply over a 1000-point
    confidence grid — see commit history if exact parity is ever needed).

    We use area range index 0 (``'all'``) and the largest max-detections cap
    (``params.maxDets[-1]``, normally 100) so the operating point lines up
    with the conditions under which mAP@0.50 is computed.
    """
    params = coco_eval.params
    target_aRng = params.areaRng[0]              # 'all'
    target_maxDet = params.maxDets[-1]           # largest cap
    iou_idx = 0                                  # IoU = 0.50

    scores: list[float] = []
    tp_flags: list[bool] = []
    total_gt = 0

    for eimg in coco_eval.evalImgs:
        if eimg is None:
            continue
        if eimg["aRng"] != target_aRng or eimg["maxDet"] != target_maxDet:
            continue

        dt_scores = np.asarray(eimg["dtScores"], dtype=np.float64)
        if dt_scores.size > 0:
            dt_matches = np.asarray(eimg["dtMatches"][iou_idx])
            dt_ignore = np.asarray(eimg["dtIgnore"][iou_idx], dtype=bool)
            keep = ~dt_ignore
            scores.extend(dt_scores[keep].tolist())
            tp_flags.extend((dt_matches[keep] > 0).tolist())

        gt_ignore = np.asarray(eimg["gtIgnore"], dtype=bool)
        total_gt += int((~gt_ignore).sum())

    if not scores or total_gt == 0:
        return 0.0, 0.0

    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    tp = np.asarray(tp_flags, dtype=np.int64)[order]
    tpc = np.cumsum(tp)
    fpc = np.cumsum(1 - tp)

    eps = 1e-16
    precision_curve = tpc / (tpc + fpc + eps)
    recall_curve = tpc / (total_gt + eps)
    f1 = 2.0 * precision_curve * recall_curve / (precision_curve + recall_curve + eps)

    best = int(np.argmax(f1))
    return float(precision_curve[best]), float(recall_curve[best])


def evaluate(
    predictions: dict[int, Prediction],
    bundle: CocoBundle,
) -> EvalResult:
    """Compute Precisão / Recall / mAP50 / mAP50-95 for a single model."""
    coco_results = predictions_to_coco_results(predictions, bundle.class_idx_to_cat_id)

    empty = EvalResult(precision=0.0, recall=0.0, map50=0.0, map50_95=0.0)

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
    precision, recall = _precision_recall_at_best_f1(coco_eval)

    return EvalResult(
        precision=precision,
        recall=recall,
        map50=map50,
        map50_95=map50_95,
    )


def write_coco_results_json(coco_results: list[dict], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(coco_results, handle)
