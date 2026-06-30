# Ensemble Method — Weighted Box Fusion for Polyp Detection

A reproducible inference + evaluation pipeline that runs three object-detection
models on the same COCO test split, fuses their predictions with
**Weighted Box Fusion (WBF)**, and reports a single, comparable metrics table
for each standalone model and the ensemble.

The three models fused are:

| Key | Model | Checkpoint |
| --- | --- | --- |
| `rfdetr` | RFDETR **nano** | `RFDETR/rfdetr_nano.pth` |
| `yolo` | YOLOv12 **nano** | `YOLO_model/yolo12n_ZScan2_ZScanKCLG_normais-80_10_10.pt` |
| `deimv2` | DEIMv2 **pico** | `DEIMv2/deimv2_pico.pth` |

The pipeline loads one model at a time, runs inference over the whole split,
evaluates it with a single unified COCO evaluator, unloads it, and finally
fuses all per-model predictions into the ensemble. Serializing the models keeps
peak VRAM bounded by the largest single model (developed on an RTX 5080, 16 GB).

---

## Table of Contents

- [How it works](#how-it-works)
- [Repository structure](#repository-structure)
- [Requirements & setup](#requirements--setup)
- [Quick start](#quick-start)
- [Configuration](#configuration)
  - [Resolution order](#resolution-order)
  - [CLI flags](#cli-flags)
  - [YAML config files & profiles](#yaml-config-files--profiles)
- [Bayesian hyperparameter optimization](#bayesian-hyperparameter-optimization)
- [Outputs](#outputs)
- [Evaluation & threshold semantics](#evaluation--threshold-semantics)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## How it works

```
load COCO split ─► for each model: load ─► infer (batched) ─► eval ─► unload
                                                                │
                                              per-model predictions
                                                                │
                                                  Weighted Box Fusion
                                                                │
                                          evaluate ENSEMBLE ─► write artifacts
```

1. **Load** the COCO test split (`_annotations.coco.json` + images).
2. **Per model**, an adapter normalizes that model's output into a shared
   `Prediction` (pixel-space `xyxy`, `scores`, `class_ids`). Inference runs in
   batches; predictions are written to `predictions_<model>.json` and scored.
3. **Fuse** every image's per-model boxes with `weighted_boxes_fusion`
   (normalized to `[0, 1]`, fused, denormalized back to pixels).
4. **Evaluate** the ensemble with the same evaluator and write the summary.

All metrics in `summary.csv` come from one source of truth — pycocotools
`COCOeval` for mAP, plus a continuous best-F1 operating point for
precision/recall (see [Evaluation & threshold semantics](#evaluation--threshold-semantics)).

---

## Repository structure

```
new_ensemble/
├── main.py                     # CLI entrypoint for the ensemble pipeline
├── bayesian_optimization.py    # Optuna TPE search over WBF/inference params
├── docker-compose.yml          # Postgres study DB + Optuna dashboard
├── requirements.txt            # Pipeline runtime dependencies
├── configs/                    # YAML config "profiles" (see below)
│   ├── example_config.yaml     # Documented template (all keys commented)
│   ├── aggressive.yaml         # High recall / low precision profile
│   ├── conservative.yaml       # Optuna-tuned, precision-leaning profile
│   ├── optuna_best.yaml        # Latest Optuna winner (auto-written)
│   └── ...
├── ensemble/                   # The pipeline package
│   ├── cli.py                  # Argument parser + RunConfig builder
│   ├── config.py               # Defaults, paths, ModelSpec, RunConfig dataclass
│   ├── config_file.py          # Strict YAML loader + override resolution
│   ├── data.py                 # COCO loading & ground-truth targets
│   ├── pipeline.py             # Orchestration: load→infer→fuse→eval→write
│   ├── fusion.py               # Weighted Box Fusion wrapper
│   ├── metrics.py              # Unified pycocotools evaluator (P/R/mAP)
│   ├── visualize.py            # Per-image & combined overlay rendering
│   └── adapters/               # Per-model wrappers → shared Prediction
│       ├── base.py             # Adapter protocol + Prediction container
│       ├── rfdetr_adapter.py
│       ├── yolo_adapter.py
│       └── deimv2_adapter.py
├── tests/                      # Pytest suite (config/CLI/pipeline/optuna)
├── DEIMv2/  RFDETR/  YOLO_model/   # Upstream model repos (local, gitignored)
└── .outputs/                   # All run artifacts (gitignored)
```

The three upstream model directories (`DEIMv2/`, `RFDETR/`, `YOLO_model/`) are
consumed as **local dependencies** and are gitignored. They must be present as
siblings of `main.py` (or located via a fallback — see
[Troubleshooting](#troubleshooting)).

---

## Requirements & setup

**Hardware:** a CUDA GPU is the default device (`cuda:0`). Developed on an
RTX 5080 (16 GB). Override with `--device cpu` or another device if needed.

**Dependencies** are pinned in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key packages: `ensemble-boxes` (WBF), `pycocotools` (evaluation),
`supervision` (data/targets), `torch`/`torchvision`, `optuna` +
`optuna-dashboard` + `psycopg2-binary` (for the hyperparameter search).

**Model backends** are installed/consumed from the sibling repositories:

- `rfdetr` — editable install of `RFDETR/`
- `ultralytics` — editable install of `YOLO_model/`
- DEIMv2 — consumed as a source tree (via `sys.path`), not pip-installed

**Pretrained weights** live *inside* each model repo (not a shared folder):

```
RFDETR/rfdetr_nano.pth
YOLO_model/yolo12n_ZScan2_ZScanKCLG_normais-80_10_10.pt
DEIMv2/deimv2_pico.pth
DEIMv2/deimv2_hgnetv2_pico_coco.yml   # DEIMv2 model config
```

**Dataset** — a COCO-format split with a single `polyp` category. The default
test split is:

```
/run/media/luizlima/ED_NVME/Desktop/Coding/ZSCAN/datasets/folders/COCO/ZScan2_ZScanKCLG_normais/test/
  ├── _annotations.coco.json
  └── *.jpg
```

Point at any other split with `--dataset /path/to/split` (the split directory
must contain `_annotations.coco.json`).

---

## Quick start

Run the full ensemble with built-in defaults:

```bash
python main.py
```

Common variations:

```bash
# Pick device / batch size
python main.py --device cuda:0 --batch-size 8

# Run a tuned profile (config file)
python main.py --config configs/conservative.yaml

# Evaluate on a different split
python main.py --dataset /path/to/COCO/.../test

# Faster smoke run: skip overlay rendering
python main.py --no-visualizations
```

On startup a banner prints exactly which parameters were overridden and where
each value came from (`[yaml]` / `[cli]`). The command prints the path to the
written `summary.csv` when it finishes. See `python main.py --help` for the
complete flag list.

---

## Configuration

Every tunable value can be set three ways. The configuration surface is in
**full parity** between the CLI flags and the YAML keys.

### Resolution order

```
config.py default   ->   --config YAML file   ->   explicit CLI flag
(lowest priority)                                   (highest priority — always wins)
```

A YAML config supplies overrides; an explicit CLI flag still beats the file.
Validation is **strict**: an unknown YAML key or a wrong-type value aborts the
run *before* any inference (with a "did you mean…" hint for typos).

### CLI flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--config` | — | YAML config file. A bare name resolves under `configs/`. |
| `--dataset` | ZScan test split | COCO split dir (must hold `_annotations.coco.json`). |
| `--output-dir` | `.outputs` | Root for run artifacts. |
| `--device` | `cuda:0` | Torch device. |
| `--batch-size` | `8` | Inference batch size. |
| `--predict-threshold` | `0.001` | Per-model inference score threshold (keep tiny — see below). |
| `--wbf-iou` | `0.7` | IoU above which WBF treats boxes as the same object. |
| `--wbf-skip-box` | `0.5` | Drop boxes below this score *inside* the fusion call. |
| `--yolo-iou` | `0.75` | Ultralytics NMS IoU for YOLO inference. **High = looser NMS**, more candidates survive into WBF (`0.7` = stock Ultralytics). |
| `--weights` | equal | WBF per-model weights, order: **RFDETR YOLO DEIMv2**. |
| `--skip-models` | none | Skip standalone model(s) (`rfdetr`/`yolo`/`deimv2`); ensemble needs ≥ 2 active. |
| `--dynamic-metrics` | `true` | `summary.csv` model rows from this run; `false` = published `HARDCODED_METRICS`. ENSEMBLE row is always live. |
| `--no-visualizations` | — | Disable per-image overlay rendering. |
| `--visualization-count` | `20` | How many images to visualize. |
| `--run-name` | `<timestamp>_ensemble` | Override the output folder name. |
| `--log-level` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR` (also read from `.env`). |
| `--rfdetr-weights` / `--yolo-weights` / `--deimv2-weights` / `--deimv2-config` / `--deimv2-dir` | in-repo layout | Override weight/config paths. |

### YAML config files & profiles

`configs/example_config.yaml` is a documented template — copy it and uncomment
only the keys you want to override. The YAML keys match the CLI flag spelling
(e.g. `wbf_iou`, `wbf_skip_box`, `yolo_iou`, `weights`).

```yaml
# configs/aggressive.yaml — higher recall, lower precision
predict_threshold: 0.001
wbf_iou: 0.15
wbf_skip_box: 0.1
yolo_iou: 0.95
weights: [1.0, 1.0, 1.0]   # order: RFDETR, YOLO, DEIMv2
```

When a run is driven by a config file and no explicit `run_name` is set, the
output folder is named after the config: `{config_stem}_{timestamp}`.

Shipped profiles:

- **`example_config.yaml`** — fully-commented reference of every key/default.
- **`aggressive.yaml`** — loose NMS + low WBF thresholds → more detections.
- **`conservative.yaml` / `optuna_best.yaml`** — Optuna-tuned winners. The
  search writes the tuned `predict_threshold`, `wbf_iou`, `yolo_iou`, and
  `weights` to `configs/optuna_best.yaml` (see below); the committed profiles
  additionally set `dynamic_metrics: false` by hand so the standalone rows show
  the published reference numbers.

---

## Bayesian hyperparameter optimization

`bayesian_optimization.py` uses **Optuna (TPE sampler)** to search the
post-processing parameter space and **maximize the ensemble `mAP@50-95`**.
Each trial runs the *full* pipeline — all three models re-infer with the
trial's parameters, predictions are fused, and the ensemble is scored with the
same unified evaluator.

**Searched parameters:**

| Parameter | Range | Notes |
| --- | --- | --- |
| `wbf_iou` | `[0.01, 0.90]` | WBF match IoU |
| `yolo_iou` | `[0.30, 0.99]` | Ultralytics NMS IoU |
| `weights` (×3) | `[0.1, 3.0]` each | order: RFDETR, YOLO, DEIMv2 |

`predict_threshold` is intentionally **frozen at `0.001`** — raising it can only
truncate the PR curve and hurt mAP.

> Optimization runs on the **validation** split by default (sibling `valid/` of
> the default test dir), so the test split stays a clean held-out set.

### 1. Bring up the study database & dashboard

Trials and the live dashboard share one PostgreSQL instance from
`docker-compose.yml`:

```bash
docker compose -f docker-compose.yml up -d
# postgres on localhost:5434, optuna-dashboard on http://localhost:8080
```

### 2. Run the search

```bash
# Default: 50 trials on the validation split, study "ensemble_wbf"
python bayesian_optimization.py

# Tune trial count / wall-clock / study name
python bayesian_optimization.py --n-trials 100
python bayesian_optimization.py --timeout 3600 --study-name ensemble_wbf_v2

# Start from a base profile (searched params are still overridden per trial)
python bayesian_optimization.py --config configs/conservative.yaml
```

Useful flags: `--dataset` (split to optimize on), `--storage` (defaults to the
compose DB), `--seed` (default `42`), `--serve-port` (script's own dashboard,
default `8081`), `--no-serve` (don't launch a dashboard after the study).

The study uses `load_if_exists=True`, so re-running with the same
`--study-name` **resumes** and accumulates trials in the same DB.

### 3. Use the winner

When the study finishes the script prints the best `mAP@50-95` and params, then
writes a `main.py`-ready config to **`configs/optuna_best.yaml`** (weights in
RFDETR, YOLO, DEIMv2 order). Reproduce the winner on the held-out test split:

```bash
python main.py --config configs/optuna_best.yaml \
    --dataset /run/.../ZScan2_ZScanKCLG_normais/test
```

Watch progress live at <http://localhost:8080> (compose dashboard) or
<http://localhost:8081> (the script's own server).

---

## Outputs

Everything is written under `.outputs/<run_name>/` (gitignored). `run_name`
defaults to `<timestamp>_ensemble`, or `<config_stem>_<timestamp>` when a config
file drives the run.

```
.outputs/optuna_best_20260616-175520/
├── summary.csv               # The headline metrics table
├── predictions_rfdetr.json   # Per-model COCO-format detections
├── predictions_yolo.json
├── predictions_deimv2.json
├── predictions_ensemble.json # Fused detections
├── run.json                  # Full machine-readable run snapshot
├── PARAMETER_VALUES.txt      # Human-readable effective params + provenance
├── logs/
│   └── pipeline.log          # Full run log (level from --log-level/.env)
└── visualizations/           # Per-image overlays (unless --no-visualizations)
    ├── <img>_rfdetr.jpg
    ├── <img>_yolo.jpg
    ├── <img>_deimv2.jpg
    ├── <img>_ensemble.jpg
    └── <img>_combined.jpg     # All models + GT on one image
```

**`summary.csv`** — the deliverable, one row per model plus the ensemble:

```csv
Modelo,Precisão,Recall,MAP 50,MAP 50-95
RFDETR nano,-,-,0.911,0.703
YOLOv12 nano,0.892,0.876,0.920,0.741
DEIMv2 pico,-,-,0.838,0.648
ENSEMBLE,0.9328,0.8373,0.8785,0.7066
```

When `dynamic_metrics: false`, the standalone rows show published reference
numbers (`HARDCODED_METRICS`); the ENSEMBLE row is **always** computed from the
current run.

**`PARAMETER_VALUES.txt`** records every effective parameter as
`name = value [source]` where source is `default`/`yaml`/`cli` — so any result
folder is self-documenting and reproducible. **`run.json`** is the same
information plus dataset stats (image/class counts, class names) in JSON.

---

## Evaluation & threshold semantics

The pipeline keeps **two distinct thresholds** — do not conflate them:

- **Predict / score threshold** (`predict_threshold`, default `0.001`): applied
  during per-model inference. Kept tiny so the full precision-recall curve
  survives for mAP. Matches what the upstream repos use.
- **Fusion `skip_box_thr`** (`wbf_skip_box`): a WBF-internal filter applied
  *inside* `weighted_boxes_fusion()`. Independent of the inference threshold.

All `summary.csv` numbers come from one evaluator (`ensemble/metrics.py`):

- **mAP@50** and **mAP@[50:95]** from pycocotools `COCOeval`.
- **Precision / Recall** from a *continuous* best-F1 operating point on the
  IoU=0.50 PR curve — reconstructed from the raw per-detection TP/FP rather than
  the binned 101-point grid, so Recall is not silently quantized. This matches
  Ultralytics' `box.mp` / `box.mr` convention.

The native evaluators inside each upstream repo remain runnable for cross-checks
but are **not** the source of the CSV numbers.

---

## Testing

The pytest suite covers config/CLI resolution, the YAML loader, pipeline
config wiring, summary/parameter output, and the Optuna objective:

```bash
pytest                       # full suite
pytest tests/test_config_file.py -q   # a single module
```

---

## Troubleshooting

**"Could not locate the upstream model directories."**
The pipeline needs `DEIMv2/`, `RFDETR/`, `YOLO_model/` reachable. Resolution
order:

1. Siblings of `main.py` (normal case).
2. A `.repo_root` text file at the project root containing the absolute path to
   the directory that holds them (useful when running from a git worktree where
   symlinks aren't possible).
3. The `ENSEMBLE_REPO_ROOT` environment variable.

**Log level** — set `LOG_LEVEL` in the project `.env` (e.g. `LOG_LEVEL=DEBUG`)
or via the process env / `--log-level`. The process env wins over `.env`.

**Optuna can't connect to the DB** — make sure the compose stack is up
(`docker compose -f docker-compose.yml up -d`); the host script talks to the
published port `5434`.

**Out of memory** — lower `--batch-size`. Models are loaded and unloaded one at
a time, so peak VRAM is bounded by the largest single model, not all three.
