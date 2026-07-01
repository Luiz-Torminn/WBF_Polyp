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


def test_evaluate_precision_and_recall_are_not_swapped():
    # 1 TP + 1 far FP, 0 FN => precision 0.5, recall 1.0 (asymmetric, so a
    # precision/recall swap in evaluate() would be caught).
    bundle = _StubBundle({1: _target([[0, 0, 10, 10]], [0])})
    preds = {1: _pred(1, [[0, 0, 10, 10], [100, 100, 110, 110]], [0.9, 0.5], [0, 0])}
    result = evaluate(preds, bundle)
    assert result.precision == 0.5
    assert result.recall == 1.0


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


# --- fixed-confidence operating point for Precision/Recall (AC1-AC7) ---------


def test_low_confidence_fp_excluded_from_precision():
    # AC1/AC5: TP@0.9 (match) + far FP@0.3 (<0.5). The FP is dropped from P/R,
    # so precision is 1.0 — but mAP still sees the FP from the full set.
    bundle = _StubBundle({1: _target([[0, 0, 10, 10]], [0])})
    preds = {1: _pred(1, [[0, 0, 10, 10], [100, 100, 110, 110]], [0.9, 0.3], [0, 0])}
    result = evaluate(preds, bundle)
    assert result.precision == 1.0
    assert result.recall == 1.0


def test_box_exactly_at_threshold_is_kept():
    # AC1: `>=` semantics — a far FP exactly at 0.5 is retained → precision 0.5.
    bundle = _StubBundle({1: _target([[0, 0, 10, 10]], [0])})
    preds = {1: _pred(1, [[0, 0, 10, 10], [100, 100, 110, 110]], [0.9, 0.5], [0, 0])}
    result = evaluate(preds, bundle)
    assert result.precision == 0.5


def test_low_confidence_tp_excluded_from_recall_but_not_map():
    # AC6/AC2: the ONLY prediction is a matching TP@0.3 (<0.5). It is filtered
    # out of Recall (recall 0.0) yet still counts toward mAP (full set), so
    # map50 stays high. This proves P/R and mAP use different prediction sets.
    bundle = _StubBundle({1: _target([[0, 0, 10, 10]], [0])})
    preds = {1: _pred(1, [[0, 0, 10, 10]], [0.3], [0])}
    result = evaluate(preds, bundle)
    assert result.recall == 0.0
    assert result.precision == 0.0
    assert result.map50 > 0.9


def test_threshold_override_includes_low_confidence_detections():
    # AC3: lowering the operating point to 0.0 restores the pre-filter behavior
    # (the FP@0.3 is counted again → precision 0.5), proving the knob works.
    bundle = _StubBundle({1: _target([[0, 0, 10, 10]], [0])})
    preds = {1: _pred(1, [[0, 0, 10, 10], [100, 100, 110, 110]], [0.9, 0.3], [0, 0])}
    result = evaluate(preds, bundle, pr_confidence_threshold=0.0)
    assert result.precision == 0.5


def test_default_operating_point_is_half():
    # AC3: the module constant documents the fixed 0.5 operating point.
    assert metrics_module.PR_CONFIDENCE_THRESHOLD == 0.5


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


# --- dynamic_metrics gating of the second (validation-default) pass ----------


class _FakeAdapter:
    name = "fake"

    def load(self, device):
        pass

    def unload(self):
        pass

    def infer_batch(self, batch):
        return []


def _patch_pipeline(monkeypatch, calls):
    """Stub run_pipeline's heavy collaborators; record _instantiate_adapter calls."""
    import ensemble.pipeline as pl
    from types import SimpleNamespace

    from ensemble.metrics import EvalResult

    fake_bundle = SimpleNamespace(image_records=[], raw={}, num_classes=1)
    monkeypatch.setattr(pl, "load_coco", lambda *a, **k: fake_bundle)
    monkeypatch.setattr(pl, "_setup_logging", lambda run: run.run_dir / "log.txt")
    monkeypatch.setattr(pl, "_serialize_run_config", lambda *a, **k: None)
    monkeypatch.setattr(
        pl, "_write_parameter_values", lambda run: run.run_dir / "PARAMETER_VALUES.txt"
    )
    monkeypatch.setattr(
        pl, "_write_summary_csv", lambda **k: k["run"].run_dir / "summary.csv"
    )
    monkeypatch.setattr(pl, "_write_visualizations", lambda **k: None)
    monkeypatch.setattr(
        pl, "_run_ensemble", lambda **k: (EvalResult(0.0, 0.0, 0.0, 0.0), {})
    )
    monkeypatch.setattr(pl, "_run_inference", lambda *a, **k: {})
    monkeypatch.setattr(pl, "evaluate", lambda *a, **k: EvalResult(0.0, 0.0, 0.0, 0.0))

    def fake_instantiate(model_key, run, overrides=None):
        calls.append((model_key, overrides))
        return _FakeAdapter()

    monkeypatch.setattr(pl, "_instantiate_adapter", fake_instantiate)


def test_dynamic_metrics_true_runs_second_validation_pass(tmp_path, monkeypatch):
    from ensemble.config import MODEL_SPECS
    from ensemble.pipeline import run_pipeline

    calls: list = []
    _patch_pipeline(monkeypatch, calls)
    run = parse_run_config(
        ["--output-dir", str(tmp_path), "--dynamic-metrics", "true"]
    )
    run.run_dir.mkdir(parents=True, exist_ok=True)
    run_pipeline(run)

    # One config pass (overrides=None) + one baseline pass (VALIDATION_DEFAULTS) per model.
    assert len(calls) == 2 * len(MODEL_SPECS)
    for spec in MODEL_SPECS:
        assert (spec.key, None) in calls
        assert (spec.key, VALIDATION_DEFAULTS[spec.key]) in calls


def test_dynamic_metrics_false_skips_second_validation_pass(tmp_path, monkeypatch):
    from ensemble.config import MODEL_SPECS
    from ensemble.pipeline import run_pipeline

    calls: list = []
    _patch_pipeline(monkeypatch, calls)
    run = parse_run_config(
        ["--output-dir", str(tmp_path), "--dynamic-metrics", "false"]
    )
    run.run_dir.mkdir(parents=True, exist_ok=True)
    run_pipeline(run)

    # Config pass only — the validation-default second pass must NOT run.
    assert len(calls) == len(MODEL_SPECS)
    assert all(overrides is None for _, overrides in calls)
