"""Tests for the ensemble evaluation operating-point decoupling.

``_run_ensemble`` reports Precision/Recall at ``wbf_skip_box_thr`` (a fair
operating point, comparable to the standalone rows) while keeping mAP over the
full PR curve (``score_threshold=0.0``) so the Optuna objective is unaffected.

The scenario is built so WBF emits one fused box ABOVE ``wbf_skip_box_thr`` (a
true positive agreed by both models) and one fused box BELOW it (a spurious
single-model box whose ``conf_type='avg'`` rescaling drops it under the
threshold). That makes the two operating points genuinely differ, so a
regression back to a single ``score_threshold=0.0`` evaluation is detectable.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from ensemble.adapters.base import Prediction
from ensemble.cli import parse_run_config
from ensemble.config import MODEL_SPECS
from ensemble.data import load_coco
from ensemble.metrics import evaluate
from ensemble.pipeline import ModelRunResult, _run_ensemble


def _write_coco(tmp_path: Path) -> Path:
    # One 100x100 image, single class (cat id 1 -> class_idx 0). Two GT boxes:
    # one that both models find (fused above threshold) and one that only a
    # single model finds (fused below threshold), so the operating point moves
    # both P/R and mAP.
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
                    {"id": 2, "image_id": 1, "category_id": 1, "bbox": [50, 50, 10, 10]},
                ],
                "categories": [{"id": 1, "name": "polyp"}],
            }
        ),
        encoding="utf-8",
    )
    return path


def _prediction(boxes, scores) -> Prediction:
    return Prediction(
        image_id=1,
        xyxy=np.array(boxes, dtype=np.float32).reshape(-1, 4),
        scores=np.array(scores, dtype=np.float32),
        class_ids=np.zeros(len(scores), dtype=np.int64),
    )


def test_ensemble_decouples_pr_threshold_from_full_curve_map(tmp_path):
    bundle = load_coco(_write_coco(tmp_path), tmp_path)

    run = dataclasses.replace(
        parse_run_config(["--output-dir", str(tmp_path)]),
        wbf_weights=(1.0, 1.0),
        wbf_iou=0.5,
        wbf_skip_box_thr=0.5,
        run_name="ensemble_decouple_test",
    )
    run.run_dir.mkdir(parents=True, exist_ok=True)
    active_specs = list(MODEL_SPECS[:2])

    # Model A finds all three boxes; model B only agrees on GT_A. After WBF
    # (conf_type='avg', weights sum 2), a box seen by both models keeps ~0.9
    # (above 0.5) while a single-model box is rescaled to 0.9 * 1/2 = 0.45
    # (below 0.5). So at the fused output:
    #   - GT_A [0,0,10,10]   : ~0.90  -> true positive ABOVE threshold
    #   - GT_B [50,50,60,60] : ~0.45  -> true positive BELOW threshold
    #   - FP   [80,80,90,90] : ~0.45  -> false positive BELOW threshold
    # The below-threshold FP makes precision move with the operating point; the
    # below-threshold TP makes mAP move too (truncating at 0.5 drops its recall).
    model_results = {
        active_specs[0].key: ModelRunResult(
            spec=active_specs[0],
            predictions={
                1: _prediction(
                    [[0, 0, 10, 10], [50, 50, 60, 60], [80, 80, 90, 90]],
                    [0.9, 0.9, 0.9],
                )
            },
            metrics=None,
            coco_results_path=Path("unused.json"),
        ),
        active_specs[1].key: ModelRunResult(
            spec=active_specs[1],
            predictions={1: _prediction([[0, 0, 10, 10]], [0.9])},
            metrics=None,
            coco_results_path=Path("unused.json"),
        ),
    }

    ensemble_metrics, ensemble_predictions, _ = _run_ensemble(
        run=run,
        bundle=bundle,
        model_results=model_results,
        active_specs=active_specs,
    )

    pr_ref = evaluate(ensemble_predictions, bundle, score_threshold=run.wbf_skip_box_thr)
    map_ref = evaluate(ensemble_predictions, bundle, score_threshold=0.0)

    # Discrimination guards: the two operating points must differ on BOTH the
    # precision axis and the mAP axis, otherwise the assertions below are vacuous
    # (a regression truncating either metric to the wrong threshold would slip
    # through). The below-threshold FP moves precision; the below-threshold TP
    # moves mAP.
    assert pr_ref.precision != pytest.approx(map_ref.precision)
    assert pr_ref.precision == pytest.approx(1.0)  # FP filtered out at 0.5
    assert map_ref.precision < pr_ref.precision  # FP counted at 0.0
    assert map_ref.map50 != pytest.approx(pr_ref.map50)  # sub-threshold TP moves AP
    assert map_ref.map50 > pr_ref.map50  # full curve recovers the sub-threshold TP

    # Precision/Recall come from the wbf_skip_box_thr evaluation...
    assert ensemble_metrics.precision == pytest.approx(pr_ref.precision)
    assert ensemble_metrics.recall == pytest.approx(pr_ref.recall)
    # ...while mAP stays on the full PR curve (the Optuna objective). Because
    # map_ref.map50 != pr_ref.map50 here, this assertion is load-bearing: a
    # regression that truncated mAP to wbf_skip_box_thr would fail it.
    assert ensemble_metrics.map50 == pytest.approx(map_ref.map50)
    assert ensemble_metrics.map50_95 == pytest.approx(map_ref.map50_95)
