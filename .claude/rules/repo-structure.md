## Repository Structure Rules

> Rules 21–24.

### Rule 21 — Respect the Project Layout

This is a single-package Python project for an object-detection ensemble, not a
JS/TS monorepo. The pipeline lives in the `ensemble/` package and is driven from
the repository root.

```txt
main.py                  → CLI entrypoint for the ensemble pipeline
bayesian_optimization.py → Optuna TPE search over WBF/inference params
ensemble/                → pipeline package (adapters, fusion, metrics, data, ...)
configs/                 → YAML config "profiles"
tests/                   → pytest suite
.outputs/                → run artifacts (gitignored)
docker-compose.yml       → Postgres study DB + Optuna dashboard
```

Place every command, dependency, and file change in the correct location for
this layout.

### Rule 22 — Upstream Model Repos Are Local Dependencies

The three model backends live as sibling directories of `main.py` and are
gitignored (consumed as editable/local dependencies, never committed):

```txt
RFDETR/       → RFDETR nano   (rfdetr editable install + rfdetr_nano.pth)
YOLO_model/   → YOLOv12 nano  (ultralytics editable install + .pt weights)
DEIMv2/       → DEIMv2 pico   (source tree via sys.path + deimv2_pico.pth)
```

They must be reachable at run time. When they are not siblings of `main.py`
(e.g. while working from a git worktree where symlinks are unavailable), point
to them with a `.repo_root` file at the project root or the
`ENSEMBLE_REPO_ROOT` environment variable.

### Rule 23 — Use pip / Conda, Not pnpm

This is a Python project. Manage dependencies with pip against
`requirements.txt`, inside the project's conda environment (see the Environment
rules). Do not use `pnpm`, `npm`, `yarn`, or any Node tooling.

```bash
pip install -r requirements.txt
```

### Rule 24 — Run Through the Python Entrypoints

Run the pipeline and experiments through the Python entrypoints, not a task
runner like Turbo:

```bash
python main.py                              # full ensemble run
python main.py --config configs/<profile>.yaml
python bayesian_optimization.py             # hyperparameter search
```
