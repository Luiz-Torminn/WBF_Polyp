"""Bayesian optimization of the ensemble's WBF / inference hyperparameters.

Optuna (TPE sampler) searches the post-processing parameter space to maximize
the ENSEMBLE ``mAP@50-95`` reported by the existing pipeline. Each trial runs
the *full* pipeline — all three models re-infer with the trial's parameters,
their predictions are fused with Weighted Box Fusion, and the ensemble is
scored with the unified COCO evaluator (``ensemble.metrics``).

Tuned parameters (``predict_threshold`` is intentionally **frozen** at 0.001 so
the full PR curve survives for mAP — raising it can only truncate the curve):

    wbf_iou        in [0.01, 0.90]
    wbf_skip_box   in [0.01, 0.90]  (log scale)
    yolo_iou       in [0.30, 0.99]
    weights        three independent values in [0.1, 3.0],
                   order RFDETR, YOLO, DEIMv2

Optimization runs against the **validation** split by default so the test split
stays a clean held-out set; rerun the winning config on test/ afterwards:

    python main.py --config configs/optuna_best.yaml \
        --dataset .../ZScan2_ZScanKCLG_normais/test

Storage is the PostgreSQL instance from ``docker-compose.optuna.yml``. Bring it
up first so trials and the live dashboard share one database::

    docker compose -f docker-compose.optuna.yml up -d   # db :5434, dashboard :8080
    python bayesian_optimization.py                     # writes trials to postgres

Usage::

    python bayesian_optimization.py --n-trials 50
    python bayesian_optimization.py --timeout 3600 --study-name ensemble_wbf_v2
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import optuna
import yaml
from optuna.samplers import TPESampler

from ensemble.cli import parse_run_config
from ensemble.config import DEFAULT_DATASET_DIR, RunConfig
from ensemble.pipeline import run_pipeline

# --- Defaults ---------------------------------------------------------------
# The DB URL targets the host-mapped port from docker-compose.optuna.yml. The
# dashboard container reaches the same DB over the docker network (5432); the
# host script must use the published port (5434).
DEFAULT_STORAGE = "postgresql+psycopg2://postgres:postgres@localhost:5434/optuna_db"
DEFAULT_STUDY_NAME = "ensemble_wbf"
DEFAULT_N_TRIALS = 50
DEFAULT_SEED = 42
# Sibling of the test split shipped in config.py; keeps tuning off the test set.
DEFAULT_DATASET = DEFAULT_DATASET_DIR.parent / "valid"
BEST_CONFIG_PATH = Path("configs/optuna_best.yaml")

# Frozen: the pipeline keeps this tiny on purpose so the full precision/recall
# curve is available for mAP integration (see ensemble/config.py).
FROZEN_PREDICT_THRESHOLD = 0.001


def build_objective(base_run: RunConfig, study_name: str):
    """Return an Optuna objective bound to a base RunConfig.

    Per trial we only swap the searched fields onto ``base_run`` via
    ``dataclasses.replace`` — every other setting (device, batch size, weight
    paths, dynamic_metrics, …) is inherited, so this stays correct even if new
    RunConfig fields are added later.
    """

    def objective(trial: optuna.Trial) -> float:
        wbf_iou = trial.suggest_float("wbf_iou", 0.01, 0.90)
        wbf_skip_box = trial.suggest_float("wbf_skip_box", 0.01, 0.90, log=True)
        yolo_iou = trial.suggest_float("yolo_iou", 0.30, 0.99)
        weights = (
            trial.suggest_float("weight_rfdetr", 0.1, 3.0),
            trial.suggest_float("weight_yolo", 0.1, 3.0),
            trial.suggest_float("weight_deimv2", 0.1, 3.0),
        )

        run = dataclasses.replace(
            base_run,
            predict_threshold=FROZEN_PREDICT_THRESHOLD,
            wbf_iou=wbf_iou,
            wbf_skip_box_thr=wbf_skip_box,
            yolo_iou_threshold=yolo_iou,
            wbf_weights=weights,
            save_visualizations=False,
            # One reused scratch dir — params/scores live in postgres, so the
            # per-trial artifacts on disk are disposable.
            run_name=f"optuna/{study_name}/scratch",
        )

        result = run_pipeline(run)
        return result.ensemble_metrics.map50_95

    return objective


def _base_run_config(args: argparse.Namespace) -> RunConfig:
    """Build the shared base RunConfig through the normal CLI resolution."""
    argv = ["--dataset", str(args.dataset), "--no-visualizations"]
    if args.config:
        argv += ["--config", args.config]
    return parse_run_config(argv)


def _write_best_config(study: optuna.Study, path: Path) -> None:
    """Emit the winning params as a `main.py --config`-ready YAML file."""
    p = study.best_params
    best = {
        "predict_threshold": FROZEN_PREDICT_THRESHOLD,
        "wbf_iou": p["wbf_iou"],
        "wbf_skip_box": p["wbf_skip_box"],
        "yolo_iou": p["yolo_iou"],
        "weights": [p["weight_rfdetr"], p["weight_yolo"], p["weight_deimv2"]],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Best ensemble hyperparameters from Optuna study '{study.study_name}'.\n"
        f"# Best ENSEMBLE mAP@50-95 = {study.best_value:.6f} "
        f"(trial #{study.best_trial.number}).\n"
        "# Reproduce on the test split:\n"
        f"#   python main.py --config {path} --dataset <.../test>\n"
        "# weights order: RFDETR, YOLO, DEIMv2\n"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header)
        yaml.safe_dump(best, handle, sort_keys=False, default_flow_style=False)


def _serve_dashboard(storage: str, port: int) -> None:
    """Best-effort local dashboard. The compose container already serves :8080,
    so this runs on a different port and never aborts the (long) study if the
    package is missing or the port is taken."""
    try:
        from optuna_dashboard import run_server
    except ImportError:
        print("optuna-dashboard not installed; skipping run_server "
              "(the compose container still serves http://localhost:8080).")
        return
    try:
        print(f"Serving optuna-dashboard at http://localhost:{port} "
              "(Ctrl-C to stop)...")
        run_server(storage, port=port)
    except (OSError, KeyboardInterrupt) as exc:
        print(f"run_server stopped: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayesian_optimization",
        description="Optuna TPE search over ensemble WBF/inference params, "
        "maximizing ENSEMBLE mAP@50-95.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help=f"Split to optimize on (default {DEFAULT_DATASET}).")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional base YAML config (searched params are "
                        "overridden per trial regardless).")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS,
                        help=f"Number of trials (default {DEFAULT_N_TRIALS}).")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Optional wall-clock limit in seconds.")
    parser.add_argument("--study-name", type=str, default=DEFAULT_STUDY_NAME)
    parser.add_argument("--storage", type=str, default=DEFAULT_STORAGE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--serve-port", type=int, default=8081,
                        help="Port for the script's own dashboard (default "
                        "8081; the compose container owns 8080).")
    parser.add_argument("--no-serve", action="store_true",
                        help="Do not launch run_server after the study.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    base_run = _base_run_config(args)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        load_if_exists=True,
    )
    study.optimize(
        build_objective(base_run, args.study_name),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    print(f"\nBest ENSEMBLE mAP@50-95: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    _write_best_config(study, BEST_CONFIG_PATH)
    print(f"Wrote best config: {BEST_CONFIG_PATH}")

    if not args.no_serve:
        _serve_dashboard(args.storage, args.serve_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
