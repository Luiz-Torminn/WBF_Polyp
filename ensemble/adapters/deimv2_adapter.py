"""DEIMv2 Pico adapter.

Builds the DEIMv2 model directly via ``engine.core.YAMLConfig`` so the adapter
sidesteps the train/solver path used by ``DEIMv2/train.py --test-only``.
Inference is fed by the same val transforms declared in the config (Resize to
640x640 + scale-to-[0,1]) and the postprocessor denormalizes boxes back to the
**original** image pixel space using ``orig_target_sizes = [W, H]`` per image
(confirmed by ``DEIMv2/engine/data/dataset/coco_dataset.py:174`` and
``engine/deim/postprocessor.py:55``).

Checkpoint selection follows DEIMv2's own ``load_tuning_state`` convention
(``engine/solver/_solver.py:174-177``): prefer EMA weights when present,
otherwise fall back to the ``model`` state dict.
"""

from __future__ import annotations

import logging
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from ensemble.adapters.base import Prediction, log_batch_predictions
from ensemble.data import ImageRecord

logger = logging.getLogger("ensemble.adapters.deimv2")


_DEIMV2_PATH_INJECTED = False
# The DEIMv2 config file ships with absolute __include__ paths pointing at a
# previous workspace location ("/run/.../ensemble-method/DEIMv2/..."). The
# adapter rewrites those to the actual DEIMv2 root before loading so the
# upstream repo stays untouched.
_STALE_DEIMV2_PREFIX = re.compile(
    r"/run/media/luizlima/ED_NVME/Desktop/Coding/ZSCAN/code/ensemble-method/DEIMv2"
)


def _ensure_deimv2_importable(deimv2_dir: Path) -> None:
    """Add the DEIMv2 directory to sys.path so its ``engine`` package imports."""
    global _DEIMV2_PATH_INJECTED
    if _DEIMV2_PATH_INJECTED:
        return
    deimv2_dir = Path(deimv2_dir).resolve()
    if not deimv2_dir.is_dir():
        raise FileNotFoundError(f"DEIMv2 directory not found: {deimv2_dir}")
    if str(deimv2_dir) not in sys.path:
        sys.path.insert(0, str(deimv2_dir))
    _DEIMV2_PATH_INJECTED = True


def _materialize_config(config_path: Path, deimv2_dir: Path) -> Path:
    """Return a config path whose ``__include__`` entries actually resolve.

    The DEIMv2 yaml loader (``engine/core/yaml_utils.py``) treats absolute
    include paths verbatim and resolves relative paths against the directory
    of the config that contains them. Two things can go wrong with the
    upstream config:
      * absolute paths point at a stale workspace
        (``/run/.../ensemble-method/DEIMv2/...``) that no longer exists.
      * relative paths assume the config sits inside the DEIMv2 directory.

    To stay isolated from the upstream repo, rewrite all of those entries
    to absolute paths anchored at ``deimv2_dir`` and materialize the result
    in a temp file. The temp file is only read by ``YAMLConfig`` once.
    """
    deimv2_root = Path(deimv2_dir).resolve()
    original_dir = Path(config_path).resolve().parent
    text = Path(config_path).read_text(encoding="utf-8")

    rewritten = _STALE_DEIMV2_PREFIX.sub(str(deimv2_root), text)

    # Rewrite relative includes to absolute paths anchored at the original
    # config directory (which is the DEIMv2 root in practice).
    def _abs_relative(match: re.Match) -> str:
        prefix, rel, suffix = match.group(1), match.group(2), match.group(3)
        absolute = (original_dir / rel).resolve()
        return f"{prefix}{absolute}{suffix}"

    rewritten = re.sub(
        r"(['\"])(\.{1,2}/[^'\"]+)(['\"])",
        _abs_relative,
        rewritten,
    )

    if rewritten == text:
        return Path(config_path)

    tmp_dir = Path(tempfile.gettempdir()) / "ensemble_deimv2_runtime"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / Path(config_path).name
    tmp_path.write_text(rewritten, encoding="utf-8")
    return tmp_path


def _filter_state_dict(model_state: dict, ckpt_state: dict) -> tuple[dict, int, int]:
    """Mirror DEIMv2 ``_solver._matched_state`` so we tolerate small differences
    between the checkpoint and the freshly built model (e.g. EMA buffers).
    """
    matched = {}
    missed = 0
    unmatched = 0
    for key, current_tensor in model_state.items():
        ckpt_tensor = ckpt_state.get(key)
        if ckpt_tensor is None:
            missed += 1
            continue
        if ckpt_tensor.shape == current_tensor.shape:
            matched[key] = ckpt_tensor
        else:
            unmatched += 1
    return matched, missed, unmatched


def _strip_module_prefix(state_dict: dict) -> dict:
    return {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


class DEIMv2Adapter:
    name = "deimv2"
    display_name = "DEIMv2 pico"

    def __init__(
        self,
        weights_path: Path,
        config_path: Path,
        deimv2_dir: Path,
        input_size: int = 640,
        score_threshold: float = 0.001,
    ):
        self._weights_path = Path(weights_path)
        self._config_path = Path(config_path)
        self._deimv2_dir = Path(deimv2_dir)
        self._input_size = int(input_size)
        self._score_threshold = float(score_threshold)
        self._model = None
        self._postprocessor = None
        self._device = "cpu"
        self._first_batch_logged = False

    def load(self, device: str) -> None:
        if self._model is not None:
            return
        if not self._weights_path.is_file():
            raise FileNotFoundError(f"DEIMv2 weights not found: {self._weights_path}")
        if not self._config_path.is_file():
            raise FileNotFoundError(f"DEIMv2 config not found: {self._config_path}")

        _ensure_deimv2_importable(self._deimv2_dir)

        import torch
        from engine.core import YAMLConfig

        config_path = _materialize_config(self._config_path, self._deimv2_dir)
        cfg = YAMLConfig(str(config_path))
        # The HGNetv2 backbone normally tries to download pretrained weights at
        # init; for inference from our own checkpoint we don't want that side
        # effect. Matches the pattern in ``DEIMv2/train.py:46-48``.
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        model = cfg.model
        postprocessor = cfg.postprocessor

        state = torch.load(str(self._weights_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "ema" in state and isinstance(state["ema"], dict):
            raw = state["ema"].get("module", state["ema"])
        elif isinstance(state, dict) and "model" in state:
            raw = state["model"]
        else:
            raw = state
        raw = _strip_module_prefix(raw)

        matched, missed, unmatched = _filter_state_dict(model.state_dict(), raw)
        model.load_state_dict(matched, strict=False)
        # `print` lands in the pipeline log; warning-level message stays out of
        # band so a clean run only emits info logs.
        if missed or unmatched:
            print(
                f"[DEIMv2Adapter] loaded {len(matched)} tensors "
                f"(missed={missed}, shape-mismatched={unmatched})"
            )

        model = model.to(device).eval()
        postprocessor = postprocessor.to(device).eval()

        self._model = model
        self._postprocessor = postprocessor
        self._device = device
        self._first_batch_logged = False
        logger.debug(
            "loaded weights=%s device=%s input_size=%d score_threshold=%.4f",
            self._weights_path,
            device,
            self._input_size,
            self._score_threshold,
        )

    def infer_batch(self, batch: list[ImageRecord]) -> list[Prediction]:
        if self._model is None or self._postprocessor is None:
            raise RuntimeError("DEIMv2Adapter.load() must be called before infer_batch")

        import torch
        from torchvision.transforms.functional import to_tensor

        tensors: list[torch.Tensor] = []
        orig_wh: list[tuple[int, int]] = []
        for record in batch:
            with Image.open(record.path) as raw_img:
                pil_img = raw_img.convert("RGB")
                width, height = pil_img.size
                resized = pil_img.resize(
                    (self._input_size, self._input_size), Image.BILINEAR
                )
                # ``ConvertPILImage(dtype='float32', scale=True)`` in DEIMv2 maps
                # uint8 → float32 in [0, 1]; ``to_tensor`` does the same plus the
                # HWC->CHW transpose.
                tensors.append(to_tensor(resized))
            orig_wh.append((width, height))

        device = self._device
        x = torch.stack(tensors, dim=0).to(device)
        # Postprocessor convention: [W, H] per row (see file docstring).
        orig_target_sizes = torch.tensor(
            [[w, h] for (w, h) in orig_wh], dtype=torch.float32, device=device
        )

        with torch.inference_mode():
            outputs = self._model(x, targets=None)
            results = self._postprocessor(outputs, orig_target_sizes)

        predictions: list[Prediction] = []
        for record, result in zip(batch, results):
            boxes = result["boxes"].detach().cpu().numpy().astype(np.float32, copy=False)
            scores = result["scores"].detach().cpu().numpy().astype(np.float32, copy=False)
            labels = result["labels"].detach().cpu().numpy().astype(np.int64, copy=False)

            if self._score_threshold > 0.0 and boxes.size:
                keep = scores >= self._score_threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            if boxes.size == 0:
                predictions.append(Prediction.empty(record.image_id))
                continue

            predictions.append(
                Prediction(
                    image_id=record.image_id,
                    xyxy=boxes,
                    scores=scores,
                    class_ids=labels,
                )
            )

        log_batch_predictions(
            "deimv2",
            predictions,
            include_samples=not self._first_batch_logged,
        )
        self._first_batch_logged = True
        return predictions

    def unload(self) -> None:
        if self._model is None and self._postprocessor is None:
            return
        try:
            import torch

            self._model = None
            self._postprocessor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            self._model = None
            self._postprocessor = None
