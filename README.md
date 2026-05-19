# Object Detection WBF Ensemble

CLI-driven object detection evaluation pipeline that runs `RFDETR nano`, `YOLOv12 nano`, and `DEIMv2 pico` on the same dataset split, converts their predictions into one shared format, applies Weighted Box Fusion, and writes standalone plus ensemble metrics under `.outputs/`.

This repository is for reproducible local experimentation, not for serving or backend APIs.

For deeper implementation details, design decisions, and model-specific caveats, see [ENSEMBLE_METHOD_PRD.md](./ENSEMBLE_METHOD_PRD.md).

## What It Does

- Loads the three local pretrained detectors.
- Runs native standalone evaluation for each model.
- Generates shared per-image predictions for ensemble fusion.
- Applies Weighted Box Fusion to compatible boxes.
- Evaluates the fused output.
- Writes a summary CSV in the format:

```text
Modelo,Precisão,Recall,MAP 50,MAP 50-95
```

## Repository Role

Project-owned code lives mainly in:

- `main.py`: CLI entrypoint.
- `ensemble_pipeline/config.py`: runtime path and threshold resolution.
- `ensemble_pipeline/orchestrator.py`: end-to-end experiment flow.
- `ensemble_pipeline/adapters/`: per-model wrappers that normalize predictions.
- `ensemble_pipeline/native_eval_*.py`: standalone native evaluators.
- `ensemble_pipeline/fusion.py`: Weighted Box Fusion integration.
- `ensemble_pipeline/evaluation.py`: ensemble metric computation.

The local model repositories `RFDETR/`, `YOLO_model/`, and `DEIMv2/` are consumed as upstream inference backends.

## Requirements

- Conda environment: `/home/luizlima/miniconda3/envs/ensemble-method`
- Local checkpoints under `pretrained_weights/`
- Dataset available in COCO format
- Python dependencies from `requirements.txt`

Recommended GPU target for full runs: `cuda:0` on the local RTX 5080 machine.

## Installation

Install the project dependencies in the `ensemble-method` conda environment:

```bash
conda run -p /home/luizlima/miniconda3/envs/ensemble-method pip install -r requirements.txt
conda run -p /home/luizlima/miniconda3/envs/ensemble-method pip install -e RFDETR -e YOLO_model
```

`DEIMv2` is loaded from the local repository path and is not installed as an editable package here.

## Correct CLI Usage

Run the full evaluation with the default configuration:

```bash
conda run -p /home/luizlima/miniconda3/envs/ensemble-method python3 main.py
```

The default dataset root is `.../COCO/ZScan2_ZScanKCLG_normais`. If you need a different COCO export, pass `--dataset-root` explicitly so standalone and ensemble runs use the same annotation set.

Run explicitly on GPU:

```bash
conda run -p /home/luizlima/miniconda3/envs/ensemble-method python3 main.py --device cuda:0
```

Run a small smoke evaluation on CPU:

```bash
conda run -p /home/luizlima/miniconda3/envs/ensemble-method python3 main.py --device cpu --limit 5 --overwrite-cache
```

Force fresh artifacts instead of reusing cached predictions:

```bash
conda run -p /home/luizlima/miniconda3/envs/ensemble-method python3 main.py --overwrite-cache
```

## Important CLI Flags

- `--device`: execution device, usually `cuda:0` or `cpu`
- `--dataset-parent`: parent directory that contains the COCO export
- `--dataset-root`: explicit dataset root override
- `--outputs-dir`: output directory override; default is `.outputs/`
- `--split`: dataset split to evaluate, default `test`
- `--limit`: limit images for smoke validation
- `--overwrite-cache`: regenerate cached artifacts
- `--no-reuse-cached-predictions`: bypass prediction cache reuse

Threshold-related flags:

- `--confidence`: general model confidence threshold for compatibility and experimentation
- `--eval-confidence`: standalone native evaluation threshold; default is `0.001`
- `--yolo-fusion-confidence`: pre-WBF score floor for YOLO detections; default `0.45`
- `--rfdetr-fusion-confidence`: pre-WBF score floor for RF-DETR detections; default `0.45`
- `--deimv2-fusion-confidence`: pre-WBF score floor for DEIMv2 detections; default `0.45`
- `--iou`: per-model IoU threshold where applicable
- `--wbf-iou`: IoU used by Weighted Box Fusion
- `--wbf-skip-box`: score floor at the fusion stage only

Important: the per-model fusion thresholds are now independent from `--eval-confidence`. Their initial defaults are set to `0.45` based on the best `mAP 50-95` point observed in the existing `.outputs/wbf_*` sweep.

## Important Config Files

- `main.py`
  CLI argument definitions and top-level execution.

- `ensemble_pipeline/config.py`
  Source of truth for resolved paths, checkpoint locations, output layout, cache policy, and threshold defaults.

- `ensemble_pipeline/orchestrator.py`
  Defines the order of execution: standalone native evaluation, shared prediction generation, fusion, ensemble evaluation, and CSV writing.

- `ensemble_pipeline/native_eval_yolo.py`
  Native YOLO evaluation integration and threshold-aware cache reuse.

- `ensemble_pipeline/native_eval_rfdetr.py`
  Native RFDETR evaluation integration.

- `ensemble_pipeline/native_eval_deimv2.py`
  Native DEIMv2 evaluation integration.

- `ENSEMBLE_METHOD_PRD.md`
  Detailed engineering reference for architecture, metric provenance, threshold semantics, output artifacts, and model-specific limitations.

## Output Artifacts

All generated files go to `.outputs/`, which is gitignored.

Key outputs:

- `.outputs/object_detection_wbf_evaluation_summary.csv`
- `.outputs/cache/run-metadata.json`
- `.outputs/cache/predictions/<model>/<image_id>.json`
- `.outputs/cache/fused/<image_id>.json`
- `.outputs/rfdetr_nano/`
- `.outputs/yolov12_nano/`
- `.outputs/deimv2_pico/`

## Threshold Semantics

The runtime intentionally separates evaluation thresholding from fusion thresholding:

- `eval_confidence`: threshold used by the standalone native evaluators
- `yolo_fusion_confidence`, `rfdetr_fusion_confidence`, `deimv2_fusion_confidence`: per-model filters applied before a model's detections enter WBF
- `wbf_skip_box`: later-stage filter applied inside WBF after the per-model prediction streams have already been filtered

Do not treat `wbf_skip_box` as a substitute for the per-model fusion thresholds. They act at different stages.

The current `0.45` initial fusion defaults are not evidence that all three models truly want different score floors. They are a conservative shared starting point derived from the existing global score-floor sweep, where `0.45` delivered the strongest `MAP 50-95` while keeping precision and recall reasonably balanced.

## Ensemble Metric Semantics

The `ENSEMBLE` row is computed by the project-owned evaluator in `ensemble_pipeline/evaluation.py`.

- `MAP 50-95`: canonical COCO `AP@[0.50:0.95 | area=all | maxDets=100]`
- `MAP 50`: canonical COCO `AP@[0.50 | area=all | maxDets=100]`
- `Precisão` and `Recall`: project-specific point metrics computed with greedy one-to-one matching at `IoU >= 0.5`, counting all ensemble predictions that enter evaluation

That means the ensemble `Precisão` and `Recall` fields are intentionally not the same metric family as COCO AP. A large low-score tail can reduce the reported `Precisão` sharply while `MAP 50` remains comparatively strong.

## Validation

Run the test suite with:

```bash
conda run -p /home/luizlima/miniconda3/envs/ensemble-method pytest -q
```

For a quick end-to-end sanity check:

```bash
conda run -p /home/luizlima/miniconda3/envs/ensemble-method python3 main.py --device cpu --limit 1 --overwrite-cache --outputs-dir .outputs/smoke
```

## Per-Model Fusion Sweep Helper

To sweep one model's pre-WBF fusion threshold using an existing cached run:

```bash
conda run -p /home/luizlima/miniconda3/envs/ensemble-method \
  python3 scripts/sweep_fusion_threshold.py \
  --base-run-dir .outputs/wbf_045 \
  --model yolo \
  --thresholds 0.10 0.20 0.30 0.40 0.45 0.50 \
  --output-dir .outputs/threshold_sweeps/yolo
```

This helper reuses cached per-model predictions, reruns only filtering plus WBF plus ensemble evaluation, and writes:

- a CSV with per-threshold metrics
- a short Markdown summary with the best `MAP 50-95` and F1 points

## Notes

- `.outputs/` is ignored by git.
- The local model repositories and pretrained weights are also ignored by git.
- Standalone model rows and the ensemble row do not come from the same evaluator implementation; see `ENSEMBLE_METHOD_PRD.md` before drawing detailed metric conclusions.
# WBF_Polyp
