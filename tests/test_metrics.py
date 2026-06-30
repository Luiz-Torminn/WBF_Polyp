"""Unit tests for the supervision-based evaluator in ensemble/metrics.py.

These use a tiny on-disk COCO file (load_coco never opens image pixels, only
builds paths) so the real ground-truth parsing path is exercised. Toy cases
cover perfect match, partial match, empty predictions, the score_threshold
operating-point filter, and the empty-ground-truth edge.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ensemble.adapters.base import Prediction
from ensemble.data import load_coco
from ensemble.metrics import EvalResult, evaluate


def _write_coco(
    tmp_path: Path,
    images: list[dict],
    annotations: list[dict],
    categories: list[dict],
) -> Path:
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            {"images": images, "annotations": annotations, "categories": categories}
        ),
        encoding="utf-8",
    )
    return path


def _prediction(image_id: int, boxes, scores, class_ids) -> Prediction:
    return Prediction(
        image_id=image_id,
        xyxy=np.array(boxes, dtype=np.float32).reshape(-1, 4),
        scores=np.array(scores, dtype=np.float32),
        class_ids=np.array(class_ids, dtype=np.int64),
    )


# Two images, two categories (cat ids 1 and 3 -> class_idx 0 and 1) so the test
# also confirms predictions use the same 0-indexed class_idx space as targets.
def _bundle_two_classes(tmp_path: Path):
    images = [
        {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
        {"id": 2, "file_name": "b.jpg", "width": 100, "height": 100},
    ]
    # COCO bbox is [x, y, w, h]; _build_targets converts to xyxy [x, y, x+w, y+h].
    annotations = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
        {"id": 2, "image_id": 2, "category_id": 3, "bbox": [5, 5, 10, 10]},
    ]
    categories = [{"id": 1, "name": "cls0"}, {"id": 3, "name": "cls1"}]
    annotations_path = _write_coco(tmp_path, images, annotations, categories)
    return load_coco(annotations_path, tmp_path)


def test_perfect_match_scores_all_ones(tmp_path):
    bundle = _bundle_two_classes(tmp_path)
    predictions = {
        1: _prediction(1, [[0, 0, 10, 10]], [0.9], [0]),
        2: _prediction(2, [[5, 5, 15, 15]], [0.8], [1]),
    }

    result = evaluate(predictions, bundle, score_threshold=0.25)

    assert isinstance(result, EvalResult)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.map50 == pytest.approx(1.0)
    assert result.map50_95 == pytest.approx(1.0)


def test_partial_match_one_image_missing(tmp_path):
    bundle = _bundle_two_classes(tmp_path)
    # Image 1 has no detection (false negative for cls0); image 2 is correct.
    predictions = {
        2: _prediction(2, [[5, 5, 15, 15]], [0.8], [1]),
    }

    result = evaluate(predictions, bundle, score_threshold=0.25)

    # WEIGHTED average over the two equally-weighted classes: cls0 = 0, cls1 = 1.
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.map50 == pytest.approx(0.5)
    assert result.map50_95 == pytest.approx(0.5)


def test_empty_predictions_scores_all_zero(tmp_path):
    bundle = _bundle_two_classes(tmp_path)

    result = evaluate({}, bundle, score_threshold=0.25)

    assert result.precision == pytest.approx(0.0, abs=1e-9)
    assert result.recall == pytest.approx(0.0, abs=1e-9)
    assert result.map50 == pytest.approx(0.0, abs=1e-9)
    assert result.map50_95 == pytest.approx(0.0, abs=1e-9)


def test_score_threshold_filters_low_confidence_detections(tmp_path):
    bundle = _bundle_two_classes(tmp_path)
    # Perfect boxes but low confidence (0.1).
    predictions = {
        1: _prediction(1, [[0, 0, 10, 10]], [0.1], [0]),
        2: _prediction(2, [[5, 5, 15, 15]], [0.1], [1]),
    }

    # Above the boxes' score -> everything filtered out -> zeros.
    filtered = evaluate(predictions, bundle, score_threshold=0.5)
    assert filtered.precision == pytest.approx(0.0, abs=1e-9)
    assert filtered.recall == pytest.approx(0.0, abs=1e-9)
    assert filtered.map50 == pytest.approx(0.0, abs=1e-9)

    # Below the boxes' score -> kept -> perfect match.
    kept = evaluate(predictions, bundle, score_threshold=0.05)
    assert kept.precision == pytest.approx(1.0)
    assert kept.recall == pytest.approx(1.0)
    assert kept.map50 == pytest.approx(1.0)


def test_empty_ground_truth_clamps_map_sentinel(tmp_path):
    images = [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}]
    categories = [{"id": 1, "name": "cls0"}]
    annotations_path = _write_coco(tmp_path, images, [], categories)
    bundle = load_coco(annotations_path, tmp_path)

    predictions = {1: _prediction(1, [[0, 0, 10, 10]], [0.9], [0])}

    result = evaluate(predictions, bundle, score_threshold=0.25)

    # Supervision returns -1 for mAP when there is no ground truth; it must be
    # clamped to 0.0, and precision is 0.0 (every detection is a false positive).
    assert result.map50 == pytest.approx(0.0, abs=1e-9)
    assert result.map50_95 == pytest.approx(0.0, abs=1e-9)
    assert result.precision == pytest.approx(0.0, abs=1e-9)
