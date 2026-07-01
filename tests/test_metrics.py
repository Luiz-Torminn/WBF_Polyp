"""Tests for supervision-based ``evaluate`` and the validation-default pass wiring."""

from __future__ import annotations

import inspect

import numpy as np
import supervision as sv

import ensemble.metrics as metrics_module
from ensemble.adapters.base import Prediction
from ensemble.cli import parse_run_config
from ensemble.config import VALIDATION_DEFAULTS
from ensemble.metrics import evaluate
from ensemble.pipeline import _instantiate_adapter


class _StubBundle:
    """Minimal stand-in for ``CocoBundle`` — ``evaluate`` only reads ``targets``."""

    def __init__(self, targets: dict[int, sv.Detections]):
        self.targets = targets


def _target(xyxy: list[list[float]], class_id: list[int]) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        class_id=np.array(class_id, dtype=np.int64),
    )


def _pred(
    image_id: int,
    xyxy: list[list[float]],
    scores: list[float],
    class_ids: list[int],
) -> Prediction:
    return Prediction(
        image_id=image_id,
        xyxy=np.array(xyxy, dtype=np.float32),
        scores=np.array(scores, dtype=np.float32),
        class_ids=np.array(class_ids, dtype=np.int64),
    )


# --- evaluate() with supervision (AC1-AC4) -----------------------------------


def test_evaluate_perfect_match_is_all_ones():
    bundle = _StubBundle({1: _target([[0, 0, 10, 10], [20, 20, 30, 30]], [0, 0])})
    preds = {1: _pred(1, [[0, 0, 10, 10], [20, 20, 30, 30]], [0.9, 0.8], [0, 0])}
    result = evaluate(preds, bundle)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.map50 == 1.0
    assert result.map50_95 == 1.0


def test_evaluate_partial_match_one_tp_one_fp_one_fn():
    # GT has 2 boxes; prediction hits one, plus a far-away FP → P=R=0.5.
    bundle = _StubBundle({1: _target([[0, 0, 10, 10], [20, 20, 30, 30]], [0, 0])})
    preds = {1: _pred(1, [[0, 0, 10, 10], [100, 100, 110, 110]], [0.9, 0.5], [0, 0])}
    result = evaluate(preds, bundle)
    assert result.precision == 0.5
    assert result.recall == 0.5
    assert 0.4 < result.map50 < 0.6


def test_evaluate_empty_predictions_is_all_zeros():
    bundle = _StubBundle({1: _target([[0, 0, 10, 10]], [0])})
    result = evaluate({}, bundle)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.map50 == 0.0
    assert result.map50_95 == 0.0


def test_evaluate_no_targets_returns_zeros():
    result = evaluate({}, _StubBundle({}))
    assert result == metrics_module.EvalResult(0.0, 0.0, 0.0, 0.0)


def test_evaluate_empty_gt_image_clamps_sentinel_to_zero():
    # An image with no ground truth would make supervision return -1; clamp to 0.
    bundle = _StubBundle({1: sv.Detections.empty()})
    preds = {1: _pred(1, [[0, 0, 10, 10]], [0.9], [0])}
    result = evaluate(preds, bundle)
    assert result.map50 >= 0.0
    assert result.map50_95 >= 0.0
    assert result.precision >= 0.0
    assert result.recall >= 0.0


def test_metrics_module_uses_supervision_not_pycocotools():
    source = inspect.getsource(metrics_module)
    assert "import supervision" in source
    # No pycocotools import (mentions in docstrings/comments are fine).
    assert "import pycocotools" not in source
    assert "from pycocotools" not in source


# --- validation-default adapter overrides (AC5) ------------------------------


def test_instantiate_adapter_applies_rfdetr_validation_defaults():
    run = parse_run_config([])
    adapter = _instantiate_adapter("rfdetr", run, VALIDATION_DEFAULTS["rfdetr"])
    assert adapter._predict_threshold == 0.001


def test_instantiate_adapter_applies_yolo_validation_defaults():
    run = parse_run_config([])
    adapter = _instantiate_adapter("yolo", run, VALIDATION_DEFAULTS["yolo"])
    assert adapter._predict_threshold == 0.001
    # iou override (0.7) must win over the config yolo_iou_threshold (0.75).
    assert adapter._iou_threshold == 0.7
    assert adapter._iou_threshold != run.yolo_iou_threshold
    assert adapter._imgsz == 640


def test_instantiate_adapter_applies_deimv2_validation_defaults():
    run = parse_run_config([])
    adapter = _instantiate_adapter("deimv2", run, VALIDATION_DEFAULTS["deimv2"])
    # score_threshold override (0.0) must win over the config predict_threshold.
    assert adapter._score_threshold == 0.0


def test_instantiate_adapter_without_overrides_uses_config():
    run = parse_run_config([])
    adapter = _instantiate_adapter("yolo", run)
    assert adapter._iou_threshold == run.yolo_iou_threshold
