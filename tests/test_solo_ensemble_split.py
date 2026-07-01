"""Tests for the solo/ensemble inference split.

The standalone (solo) rows must be a config-invariant baseline: only the
ENSEMBLE run consumes the tuned ``yolo_iou``. These exercise the mechanism that
guarantees that — the inference signature, the mode-aware adapter factory, and
the skip-solo summary — without loading any model weights or running inference.
"""

from __future__ import annotations

import csv
import dataclasses

from ensemble.cli import parse_run_config
from ensemble.config import DEFAULT_YOLO_IOU_THRESHOLD, MODEL_SPECS
from ensemble.pipeline import (
    _inference_signature,
    _instantiate_adapter,
    _write_summary_csv,
)


def _run_with_yolo_iou(yolo_iou: float):
    """A base RunConfig with the tuned ensemble YOLO IoU set to ``yolo_iou``."""
    base = parse_run_config([])
    return dataclasses.replace(base, yolo_iou_threshold=yolo_iou)


# --- inference signature -----------------------------------------------------


def test_yolo_solo_signature_is_invariant_to_config_yolo_iou():
    """Two configs differing only in yolo_iou yield the SAME YOLO solo signature
    (so the solo YOLO row is identical) but DIFFERENT ensemble signatures."""
    run_a = _run_with_yolo_iou(0.30)
    run_b = _run_with_yolo_iou(0.90)

    solo_a = _inference_signature("yolo", run_a, "solo")
    solo_b = _inference_signature("yolo", run_b, "solo")
    assert solo_a == solo_b  # config-invariant solo baseline

    ens_a = _inference_signature("yolo", run_a, "ensemble")
    ens_b = _inference_signature("yolo", run_b, "ensemble")
    assert ens_a != ens_b  # tuned param still shapes the fusion input


def test_yolo_solo_uses_native_default_iou():
    run = _run_with_yolo_iou(0.30)
    # Solo signature carries the model default, not the tuned config value.
    assert DEFAULT_YOLO_IOU_THRESHOLD in _inference_signature("yolo", run, "solo")
    assert 0.30 in _inference_signature("yolo", run, "ensemble")


def test_yolo_needs_two_passes_only_when_tuned_iou_differs_from_default():
    tuned = _run_with_yolo_iou(0.30)
    assert _inference_signature("yolo", tuned, "solo") != _inference_signature(
        "yolo", tuned, "ensemble"
    )

    at_default = _run_with_yolo_iou(DEFAULT_YOLO_IOU_THRESHOLD)
    # When the config equals the default, the two passes collapse into one.
    assert _inference_signature("yolo", at_default, "solo") == _inference_signature(
        "yolo", at_default, "ensemble"
    )


def test_rfdetr_and_deimv2_signatures_are_mode_independent():
    run = _run_with_yolo_iou(0.30)
    for key in ("rfdetr", "deimv2"):
        # No tuned inference knob -> solo and ensemble share a single pass.
        assert _inference_signature(key, run, "solo") == _inference_signature(
            key, run, "ensemble"
        )


# --- mode-aware adapter factory ----------------------------------------------


def test_instantiate_yolo_adapter_respects_mode():
    run = _run_with_yolo_iou(0.30)
    solo = _instantiate_adapter("yolo", run, "solo")
    ensemble = _instantiate_adapter("yolo", run, "ensemble")
    # YOLOAdapter stores the NMS IoU it will pass to predict().
    assert solo._iou_threshold == DEFAULT_YOLO_IOU_THRESHOLD
    assert ensemble._iou_threshold == 0.30


# --- skip-solo summary -------------------------------------------------------


def test_summary_skips_solo_rows_when_skip_solo_metrics(tmp_path):
    base = parse_run_config(["--output-dir", str(tmp_path), "--dynamic-metrics", "true"])
    run = dataclasses.replace(base, skip_solo_metrics=True)
    run.run_dir.mkdir(parents=True, exist_ok=True)

    from ensemble.metrics import EvalResult

    csv_path = _write_summary_csv(
        run=run,
        active_specs=list(MODEL_SPECS),
        model_results={},  # never indexed when solo is skipped
        ensemble_metrics=EvalResult(precision=0.9, recall=0.8, map50=0.95, map50_95=0.7),
    )
    with open(csv_path, encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    # Header + ENSEMBLE only, no per-model solo rows.
    assert len(rows) == 2
    assert rows[0][0] == "Modelo"
    assert rows[1][0] == "ENSEMBLE"
