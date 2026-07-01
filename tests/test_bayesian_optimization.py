"""Tests for the Optuna objective and best-config emitter.

These exercise the real param->RunConfig mapping (with run_pipeline mocked so no
GPU/inference runs) and verify the emitted YAML round-trips through the
pipeline's own strict config loader.
"""

from __future__ import annotations

from unittest.mock import patch

import optuna

import bayesian_optimization as bo
from ensemble.cli import parse_run_config
from ensemble.config_file import load_config_file
from ensemble.metrics import EvalResult
from ensemble.pipeline import PipelineResult

_PARAMS = {
    "wbf_iou": 0.5,
    "wbf_skip_box": 0.05,
    "yolo_iou": 0.6,
    "weight_rfdetr": 1.5,
    "weight_yolo": 2.0,
    "weight_deimv2": 0.5,
}


def test_objective_maps_params_into_runconfig(tmp_path):
    base = parse_run_config([])
    captured: dict = {}

    def fake_run_pipeline(run):
        captured["run"] = run
        return PipelineResult(
            summary_path=tmp_path / "summary.csv",
            ensemble_metrics=EvalResult(
                precision=0.1, recall=0.2, map50=0.3, map50_95=0.4242
            ),
        )

    objective = bo.build_objective(base, "study_x")
    with patch.object(bo, "run_pipeline", fake_run_pipeline):
        value = objective(optuna.trial.FixedTrial(_PARAMS))

    # The objective returns the ENSEMBLE mAP@50-95 verbatim (full precision).
    assert value == 0.4242

    run = captured["run"]
    assert run.wbf_iou == 0.5
    # wbf_skip_box is not searched (the suggest is disabled); the objective
    # freezes wbf_skip_box_thr to 0.5 regardless of any param passed in.
    assert run.wbf_skip_box_thr == 0.5
    assert run.yolo_iou_threshold == 0.6
    # Weight order must be RFDETR, YOLO, DEIMv2.
    assert run.wbf_weights == (1.5, 2.0, 0.5)
    # predict_threshold is frozen regardless of anything else.
    assert run.predict_threshold == bo.FROZEN_PREDICT_THRESHOLD == 0.001
    # Visualizations are forced off and a single scratch dir is reused.
    assert run.save_visualizations is False
    assert run.run_name == "optuna/study_x/scratch"
    # The search path skips solo work — only the ENSEMBLE metric is needed.
    assert run.skip_solo_metrics is True


def test_best_config_yaml_roundtrips_through_loader(tmp_path):
    study = optuna.create_study(direction="maximize")

    def dummy(trial):
        for name in ("wbf_iou", "yolo_iou"):
            trial.suggest_float(name, 0.01, 0.99)
        for name in ("weight_rfdetr", "weight_yolo", "weight_deimv2"):
            trial.suggest_float(name, 0.1, 3.0)
        return 0.5

    study.optimize(dummy, n_trials=1)

    out = tmp_path / "optuna_best.yaml"
    bo._write_best_config(study, out)

    # The pipeline's strict loader rejects unknown keys / bad types, so a clean
    # parse proves the emitted file is directly usable via `main.py --config`.
    loaded = load_config_file(out)
    assert loaded["predict_threshold"] == 0.001
    assert "wbf_iou" in loaded
    # wbf_skip_box is no longer searched, so it is not emitted.
    assert "wbf_skip_box_thr" not in loaded
    assert "yolo_iou_threshold" in loaded
    assert len(loaded["wbf_weights"]) == 3
