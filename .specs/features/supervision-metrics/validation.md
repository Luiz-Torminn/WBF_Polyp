# Validation Report — supervision-metrics

Date: 2026-07-01
Verifier: independent (author != verifier), evidence-or-zero.
Worktree: `../working_trees/supervision-metrics` (branch `feat/supervision-metrics`)

## Overall verdict: PASS

All 8 acceptance criteria have real, spec-matching test coverage or direct
code-inspection evidence. `pytest` is green except the 2 pre-existing,
unrelated `bayesian_optimization.py` failures. Mutation testing killed 2/4
targeted mutants; the 2 survivors are genuine, previously-flagged gaps (not
regressions) — see Gaps below.

## 1. Baseline test run

```
python -m pytest -q
...
FAILED tests/test_bayesian_optimization.py::test_objective_maps_params_into_runconfig
FAILED tests/test_bayesian_optimization.py::test_best_config_yaml_roundtrips_through_loader
2 failed, 43 passed, 2 warnings in 0.56s
```

Confirmed pre-existing on `main_working` (not caused by this feature):
`git show main_working:bayesian_optimization.py` line 74 has
`# wbf_skip_box = trial.suggest_float("wbf_skip_box", 0.01, 0.90, log=True)`
commented out, while the tests still assert `wbf_skip_box_thr` is present /
overridden. This mismatch pre-dates the feature branch. Every other test
(43/45) passes.

## 2. Spec-anchored AC coverage

| AC | Covering test(s) | Asserted vs expected | Verdict |
|----|-------------------|------------------------|---------|
| AC1 — perfect match → all 1.0 | `tests/test_metrics.py::test_evaluate_perfect_match_is_all_ones` | asserts `precision==recall==map50==map50_95==1.0` | PASS |
| AC2 — 1TP+1FP+1FN → P≈R≈0.5, map50≈0.5 | `test_evaluate_partial_match_one_tp_one_fp_one_fn` | asserts `precision==0.5`, `recall==0.5`, `0.4<map50<0.6` | PASS |
| AC3 — empty preds / empty GT → finite non-negative, -1 clamped | `test_evaluate_empty_predictions_is_all_zeros`, `test_evaluate_no_targets_returns_zeros`, `test_evaluate_empty_gt_image_clamps_sentinel_to_zero` | asserts exact 0.0 for empty-preds/no-targets cases; asserts `>=0.0` for the empty-GT-image sentinel case (mutation-confirmed to actually exercise the clamp — see 3(a)) | PASS |
| AC4 — imports supervision not pycocotools, stable `EvalResult`/`evaluate` signature | `test_metrics_module_uses_supervision_not_pycocotools` (source-inspects for `import supervision`, absence of pycocotools imports); signature verified by reading `ensemble/metrics.py:90-93` (`evaluate(predictions: dict[int, Prediction], bundle: CocoBundle) -> EvalResult`, no threshold param) | PASS |
| AC5 — validation-default override params per model | `test_instantiate_adapter_applies_rfdetr_validation_defaults` (`predict_threshold==0.001`), `test_instantiate_adapter_applies_yolo_validation_defaults` (`predict_threshold==0.001`, `iou_threshold==0.7` and `!= run.yolo_iou_threshold`, `imgsz==640`), `test_instantiate_adapter_applies_deimv2_validation_defaults` (`score_threshold==0.0`) | matches spec table exactly | PASS |
| AC6 — dynamic_metrics=False quotes HARDCODED_METRICS, no per-model validation pass | `test_summary_hardcoded_when_disabled` (CSV rows == `HARDCODED_METRICS`) + `test_ensemble_row_always_computed_even_when_disabled` (ENSEMBLE still computed) cover the CSV-quoting half. The "no second pass runs" half is **not test-exercised** (would require mocking model loading) — verified instead by direct code read: `ensemble/pipeline.py:265` `if run.dynamic_metrics:` gates the entire baseline-adapter/`_run_inference` block; the `else` branch (`pipeline.py:282-283`) only constructs a static zero `EvalResult`, no inference call. Mutation test (3(d)) confirms this gate is currently *not* covered by any test — flagged as a gap, not a failure of the implementation. | PASS (implementation correct; coverage gap noted) |
| AC7 — no `predictions_to_coco_results`/`write_coco_results_json` in `ensemble/`, pytest green | `grep -rn "predictions_to_coco_results\|write_coco_results_json" ensemble/` → only hits inside the gitignored `ensemble/metrics_old.py` (excluded by spec); zero hits in `ensemble/metrics.py`, `ensemble/pipeline.py`, or anywhere else in `ensemble/`. `grep -rn coco_results_path` → no hits anywhere (field fully removed, not just renamed). pytest: 43 passed / 2 pre-existing-unrelated failures (see §1). | PASS |
| AC8 — `metrics_old.py` exists, verbatim, gitignored, unimported | `diff <(git show main_working:ensemble/metrics.py) ensemble/metrics_old.py` → empty (byte-identical to the old tracked `metrics.py`). `git check-ignore -v ensemble/metrics_old.py` → matched by `.gitignore:48`. `git ls-files ensemble/metrics_old.py` → empty (untracked). `grep -rn metrics_old ensemble/ tests/` → only a docstring mention in `ensemble/metrics.py:9` (`` `metrics_old.py`) ``), no `import` statement anywhere. | PASS |

## 3. Discrimination sensor (mutation testing)

All mutants applied one at a time to the actual source, tests re-run, then
reverted with `git checkout -- <file>`.

| Mutant | Change | Result | Killed by |
|--------|--------|--------|-----------|
| (a) `ensemble/metrics.py::_nonneg` | `max(0.0, float(value))` → `float(value)` | **KILLED** — `test_evaluate_empty_gt_image_clamps_sentinel_to_zero` failed (`assert -1.0 >= 0.0`) | `tests/test_metrics.py` |
| (b) `ensemble/metrics.py::evaluate` | swapped `precision_at_50`/`recall_at_50` into the wrong `EvalResult` fields | **SURVIVED** — all 10 tests in `test_metrics.py` still passed | none — flagged below |
| (c) `ensemble/config.py::VALIDATION_DEFAULTS["yolo"]` | `iou_threshold` `0.7` → `0.75` | **KILLED** — `test_instantiate_adapter_applies_yolo_validation_defaults` failed (`assert 0.75 == 0.7`) | `tests/test_metrics.py` |
| (d) `ensemble/pipeline.py` model loop | `if run.dynamic_metrics:` → `if True:` (unconditional second pass) | **SURVIVED** — full suite: 43 passed / 2 pre-existing failures, no new failures | none — flagged below |

Working tree confirmed clean after all four mutate/revert cycles:
`git status --short` → empty (verified after each individual revert and
again at the end).

## 4. Diff range

```
git log main_working..HEAD --oneline
292b47b refactor(pipeline): drop COCO results JSON artifacts (unused downstream)
3221415 feat(metrics): compute P/R/mAP via supervision instead of pycocotools
4bce587 feat(pipeline): second validation-default pass for standalone rows
944c015 feat(config): add per-model validation-mode default params
7824749 chore(metrics): preserve pycocotools evaluator as gitignored metrics_old.py
c79ddf8 docs(specs): supervision-metrics spec, design, tasks, decisions
```

## 5. Ranked gap list

1. **Surviving mutant (b) — precision/recall swap undetected (spec-precision gap).**
   Because AC1 requires `precision==recall==1.0` and AC2 requires
   `precision==recall==0.5`, no existing/spec-mandated test can distinguish
   the two fields from each other. This is a real hole: a future regression
   that silently swaps precision and recall would go undetected by the test
   suite. Recommend adding one asymmetric-P/R toy case (e.g. GT with 3 boxes,
   predictions producing 2 TP + 2 FP → precision=0.5, recall=0.667) to
   `tests/test_metrics.py` to make the two fields distinguishable.
2. **Surviving mutant (d) — `dynamic_metrics` gate in `run_pipeline`'s model
   loop has no test coverage.** No test drives `run_pipeline` end-to-end
   (it requires real model loading/adapters), so the "no second pass when
   disabled" half of AC6/R8 is verified only by code reading, not by an
   executable test. This was anticipated in the design (adapters aren't
   `.load()`-ed in tests). Recommend either an integration test with fake/stub
   adapters injected via a seam, or explicitly documenting this as an
   accepted, manually-verified gap.
3. **Minor doc drift (non-blocking).** `ensemble/pipeline.py`'s module
   docstring (lines 1-6) still says "writes that model's COCO results JSON",
   which is stale after R10/AC7 removed COCO JSON writing. Cosmetic only;
   does not affect behavior or test correctness. Flagged for cleanup.

No other gaps found. All committed changes are surgical and traceable to the
spec's requirements (R1-R13); no scope creep detected in the diff range.

## 6. Gap resolution (post-verification)

All three ranked gaps were closed in commit `a2ac337` and the follow-up:

1. **RESOLVED** — added `test_evaluate_precision_and_recall_are_not_swapped`
   (1 TP + 1 FP + 0 FN → precision 0.5, recall 1.0). Mutant (b) now killed
   (re-verified: swapping P/R fails the test).
2. **RESOLVED** — added `test_dynamic_metrics_true_runs_second_validation_pass`
   and `test_dynamic_metrics_false_skips_second_validation_pass` driving
   `run_pipeline` with stubbed collaborators and counting `_instantiate_adapter`
   calls (6 when enabled, 3 when disabled). Mutant (d) now killed.
3. **RESOLVED** — pipeline module docstring no longer mentions COCO JSON.

CodeRabbit pass (`--base main_working`): 1 major finding on `evaluate()` image-id
selection was reviewed and **declined with justification** — `bundle.targets`
holds an entry for every image in the COCO manifest (empty-GT images included as
`sv.Detections.empty()`), and predictions are only ever produced over
`bundle.image_records` (the same manifest), so `set(predictions) ⊆
set(bundle.targets)` is invariant and no prediction is ever dropped. Iterating
`bundle.targets` keys is also mandated by spec R4. The two minor findings (a
`tasks.md` doc pattern and this gap list) were doc-only and fixed.
