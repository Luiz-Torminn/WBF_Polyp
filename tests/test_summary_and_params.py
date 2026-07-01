"""Tests for the dynamic-metrics summary.csv, config-based run naming, and
the PARAMETER_VALUES.txt control file."""

from __future__ import annotations

import csv
from pathlib import Path

from ensemble.cli import parse_run_config
from ensemble.config import HARDCODED_METRICS, MODEL_SPECS
from ensemble.metrics import EvalResult
from ensemble.pipeline import (
    ModelRunResult,
    _write_parameter_values,
    _write_summary_csv,
)


def _run(tmp_path, argv):
    """Build a RunConfig rooted at tmp_path and create its run_dir."""
    run = parse_run_config(["--output-dir", str(tmp_path), *argv])
    run.run_dir.mkdir(parents=True, exist_ok=True)
    return run


def _model_results() -> dict[str, ModelRunResult]:
    results: dict[str, ModelRunResult] = {}
    for i, spec in enumerate(MODEL_SPECS):
        metrics = EvalResult(
            precision=0.1 + i, recall=0.2 + i, map50=0.3 + i, map50_95=0.4 + i
        )
        results[spec.key] = ModelRunResult(
            spec=spec,
            ensemble_predictions={},
            solo_predictions={},
            metrics=metrics,
            coco_results_path=Path("unused.json"),
        )
    return results


def _read_rows(csv_path) -> list[list[str]]:
    with open(csv_path, encoding="utf-8") as handle:
        return list(csv.reader(handle))


def _ensemble_metrics() -> EvalResult:
    return EvalResult(precision=0.9, recall=0.8, map50=0.95, map50_95=0.7)


# --- summary.csv branching ---------------------------------------------------


def test_summary_dynamic_uses_computed_metrics_one_row_each(tmp_path):
    run = _run(tmp_path, ["--dynamic-metrics", "true"])
    csv_path = _write_summary_csv(
        run=run,
        active_specs=list(MODEL_SPECS),
        model_results=_model_results(),
        ensemble_metrics=_ensemble_metrics(),
    )
    rows = _read_rows(csv_path)
    # header + 3 models + ensemble, no duplicates (the old double-write bug).
    assert len(rows) == 1 + len(MODEL_SPECS) + 1
    body = {r[0]: r[1:] for r in rows[1:]}
    # rfdetr is index 0 -> precision 0.1000 (computed, not the "-" hardcoded).
    assert body["RFDETR nano"] == ["0.1000", "0.2000", "0.3000", "0.4000"]


def test_summary_hardcoded_when_disabled(tmp_path):
    run = _run(tmp_path, ["--dynamic-metrics", "false"])
    csv_path = _write_summary_csv(
        run=run,
        active_specs=list(MODEL_SPECS),
        model_results=_model_results(),
        ensemble_metrics=_ensemble_metrics(),
    )
    body = {r[0]: r[1:] for r in _read_rows(csv_path)[1:]}
    for spec in MODEL_SPECS:
        hc = HARDCODED_METRICS[spec.key]
        assert body[spec.display_name] == [
            hc["precision"],
            hc["recall"],
            hc["map50"],
            hc["map50_95"],
        ]


def test_ensemble_row_always_computed_even_when_disabled(tmp_path):
    run = _run(tmp_path, ["--dynamic-metrics", "false"])
    csv_path = _write_summary_csv(
        run=run,
        active_specs=list(MODEL_SPECS),
        model_results=_model_results(),
        ensemble_metrics=_ensemble_metrics(),
    )
    body = {r[0]: r[1:] for r in _read_rows(csv_path)[1:]}
    assert body["ENSEMBLE"] == ["0.9000", "0.8000", "0.9500", "0.7000"]


# --- config-based run naming -------------------------------------------------


def test_run_name_follows_config_stem(tmp_path):
    cfg = tmp_path / "aggressive.yaml"
    cfg.write_text("wbf_iou: 0.5\n", encoding="utf-8")
    run = parse_run_config(["--config", str(cfg)])
    assert run.run_name.startswith("aggressive_")
    assert run.run_name != "aggressive_"  # timestamp suffix present


def test_explicit_run_name_beats_config_naming(tmp_path):
    cfg = tmp_path / "aggressive.yaml"
    cfg.write_text("wbf_iou: 0.5\n", encoding="utf-8")
    run = parse_run_config(["--config", str(cfg), "--run-name", "mine"])
    assert run.run_name == "mine"


def test_yaml_run_name_beats_config_naming(tmp_path):
    cfg = tmp_path / "aggressive.yaml"
    cfg.write_text("run_name: from_yaml\n", encoding="utf-8")
    run = parse_run_config(["--config", str(cfg)])
    assert run.run_name == "from_yaml"


def test_no_config_keeps_default_run_name(tmp_path):
    run = parse_run_config([])
    assert run.run_name.endswith("_ensemble")


# --- PARAMETER_VALUES.txt ----------------------------------------------------


def test_parameter_values_records_all_fields_with_sources(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("wbf_iou: 0.5\n", encoding="utf-8")
    run = _run(tmp_path, ["--config", str(cfg), "--batch-size", "16"])

    path = _write_parameter_values(run)
    text = path.read_text(encoding="utf-8")

    assert path.name == "PARAMETER_VALUES.txt"
    assert "# Config file:" in text and "c.yaml" in text
    # every RunConfig field except `extra` is present
    assert "dynamic_metrics" in text
    assert "wbf_iou" in text
    # source provenance is rendered
    assert "[yaml]" in text  # wbf_iou
    assert "[cli]" in text  # batch_size
    assert "[default]" in text  # untouched fields
    # bool rendered YAML-style
    assert "dynamic_metrics" in text and "true" in text


def test_parameter_values_no_config_marks_builtin(tmp_path):
    run = _run(tmp_path, [])
    text = _write_parameter_values(run).read_text(encoding="utf-8")
    assert "(built-in defaults)" in text
    assert "[default]" in text
