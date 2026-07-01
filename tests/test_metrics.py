"""Tests for supervision-based ``evaluate`` and the validation-default pass wiring."""

from __future__ import annotations

from ensemble.cli import parse_run_config
from ensemble.config import VALIDATION_DEFAULTS
from ensemble.pipeline import _instantiate_adapter


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
