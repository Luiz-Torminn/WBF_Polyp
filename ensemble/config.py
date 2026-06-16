"""Runtime configuration for the ensemble pipeline.

The pipeline expects the three upstream model repositories to be reachable
under a single root directory. After the worktree branch is merged back to
``main_working`` the repositories live as siblings of ``main.py`` so the
default resolution is trivial. When running from a git worktree the upstream
repositories cannot always be symlinked (filesystem limitation), so a
``.repo_root`` text file at the project root is honored as a fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REQUIRED_MODEL_DIRS = ("DEIMv2", "RFDETR", "YOLO_model")


def _has_model_dirs(candidate: Path) -> bool:
    return all((candidate / name).is_dir() for name in _REQUIRED_MODEL_DIRS)


def _read_env_file(path: Path) -> dict[str, str]:
    """Tiny KEY=VALUE parser for the project-local ``.env`` file.

    Avoids a hard dependency on ``python-dotenv``; the file is currently a
    single line (``LOG_LEVEL=DEBUG``) so a full library is overkill.
    """
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip("'\"")
    return env


_ENV_VARS: dict[str, str] = _read_env_file(_PROJECT_ROOT / ".env")


def _resolve_log_level() -> str:
    """Process env wins over .env; default INFO matches prior behavior."""
    raw = os.environ.get("LOG_LEVEL") or _ENV_VARS.get("LOG_LEVEL") or "INFO"
    return str(raw).upper()


def find_repo_root() -> Path:
    """Locate the directory that contains DEIMv2/, RFDETR/, YOLO_model/.

    Resolution order:
      1. The project root (where ``ensemble/`` lives) — normal case after merge.
      2. A ``.repo_root`` pointer file inside the project root — worktree case.
      3. The ``ENSEMBLE_REPO_ROOT`` environment variable.
    """

    if _has_model_dirs(_PROJECT_ROOT):
        return _PROJECT_ROOT

    pointer = _PROJECT_ROOT / ".repo_root"
    if pointer.is_file():
        candidate = Path(pointer.read_text().strip()).expanduser().resolve()
        if _has_model_dirs(candidate):
            return candidate

    env_value = os.environ.get("ENSEMBLE_REPO_ROOT")
    if env_value:
        candidate = Path(env_value).expanduser().resolve()
        if _has_model_dirs(candidate):
            return candidate

    raise RuntimeError(
        "Could not locate the upstream model directories "
        f"({', '.join(_REQUIRED_MODEL_DIRS)}). Either place this project as a sibling of "
        "DEIMv2/, RFDETR/, YOLO_model/, write the absolute path of that location into "
        f"{_PROJECT_ROOT / '.repo_root'!s}, or export ENSEMBLE_REPO_ROOT."
    )


REPO_ROOT: Path = find_repo_root()

DEIMV2_DIR: Path = REPO_ROOT / "DEIMv2"
RFDETR_DIR: Path = REPO_ROOT / "RFDETR"
YOLO_DIR: Path = REPO_ROOT / "YOLO_model"

DEIMV2_WEIGHTS: Path = DEIMV2_DIR / "deimv2_pico.pth"
DEIMV2_CONFIG: Path = DEIMV2_DIR / "deimv2_hgnetv2_pico_coco.yml"

RFDETR_WEIGHTS: Path = RFDETR_DIR / "rfdetr_nano.pth"

YOLO_WEIGHTS: Path = YOLO_DIR / "yolo12n_ZScan2_ZScanKCLG_normais-80_10_10.pt"

DEFAULT_DATASET_DIR: Path = Path(
    "/run/media/luizlima/ED_NVME/Desktop/Coding/ZSCAN/datasets/folders/COCO/"
    "ZScan2_ZScanKCLG_normais/test"
)

DEFAULT_OUTPUT_DIR: Path = _PROJECT_ROOT / ".outputs"


# Threshold semantics — see AGENTS.md "Evaluation Threshold Semantics".
# The predict threshold is intentionally tiny so the full PR curve survives
# for mAP computation; the WBF skip threshold is independent and only
# filters boxes inside the fusion call.
DEFAULT_PREDICT_THRESHOLD: float = 0.001
DEFAULT_WBF_IOU: float = 0.7
DEFAULT_WBF_SKIP_BOX_THR: float = 0.5

# Ultralytics treats `iou` as the IoU ABOVE which a same-class box is
# suppressed by NMS. To minimize NMS interference before WBF (which handles
# dedup itself), the ensemble path defaults to a high value so most YOLO
# candidates survive. The literal Ultralytics default is 0.7; override with
# `--yolo-iou` to reproduce the standalone YOLO_model/main.py behavior.
DEFAULT_YOLO_IOU_THRESHOLD: float = 0.75

DEFAULT_DEVICE: str = "cuda:0"
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_VISUALIZATION_COUNT: int = 8

# When True (default) the summary.csv standalone-model rows use the metrics
# computed from this run's inference. When False they fall back to the
# published numbers in ``HARDCODED_METRICS`` below. The ENSEMBLE row is always
# computed regardless of this flag.
DEFAULT_DYNAMIC_METRICS: bool = True

# Resolved at import time from .env (LOG_LEVEL=...) or the process env.
DEFAULT_LOG_LEVEL: str = _resolve_log_level()


# Canonical display names used in the CSV ("Modelo" column) and in
# log/output artifact filenames.
@dataclass(frozen=True)
class ModelSpec:
    key: str  # filesystem/CLI safe (e.g. 'rfdetr')
    display_name: str  # CSV-facing label (e.g. 'RFDETR nano')


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(key="rfdetr", display_name="RFDETR nano"),
    ModelSpec(key="yolo", display_name="YOLOv12 nano"),
    ModelSpec(key="deimv2", display_name="DEIMv2 pico"),
)

ENSEMBLE_DISPLAY_NAME: str = "ENSEMBLE"


# Published per-model metrics used for the summary.csv standalone-model rows
# when ``dynamic_metrics`` is disabled. Keyed by ``ModelSpec.key``; the four
# values map to the CSV columns Precisão, Recall, MAP 50, MAP 50-95. A "-"
# marks a metric that is not reported for that model.
HARDCODED_METRICS: dict[str, dict[str, str]] = {
    "rfdetr": {"precision": "-", "recall": "-", "map50": "0.911", "map50_95": "0.703"},
    "yolo": {"precision": "0.892", "recall": "0.876", "map50": "0.920", "map50_95": "0.741"},
    "deimv2": {"precision": "-", "recall": "-", "map50": "0.838", "map50_95": "0.648"},
}


@dataclass(frozen=True)
class RunConfig:
    """Resolved runtime configuration for a single pipeline invocation."""

    dataset_dir: Path
    output_dir: Path
    device: str
    batch_size: int
    predict_threshold: float
    wbf_iou: float
    wbf_skip_box_thr: float
    wbf_weights: tuple[float, ...]
    skip_models: tuple[str, ...]
    save_visualizations: bool
    dynamic_metrics: bool
    visualization_count: int
    run_name: str
    rfdetr_weights: Path
    yolo_weights: Path
    deimv2_weights: Path
    deimv2_config: Path
    deimv2_dir: Path
    yolo_iou_threshold: float
    log_level: str
    extra: dict = field(default_factory=dict)

    @property
    def annotations_path(self) -> Path:
        return self.dataset_dir / "_annotations.coco.json"

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name

    @property
    def predictions_dir(self) -> Path:
        return self.run_dir

    @property
    def visualizations_dir(self) -> Path:
        return self.run_dir / "visualizations"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"
