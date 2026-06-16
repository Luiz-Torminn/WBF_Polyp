"""Tests for the YAML config-file loader, validation, merge, and banner."""

from __future__ import annotations

import pytest

from ensemble.config_file import (
    ConfigError,
    Override,
    load_config_file,
    render_banner,
    resolve,
)


def _write(tmp_path, text: str):
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_coerces_known_keys_to_runconfig_fields(tmp_path):
    path = _write(tmp_path, "wbf_iou: 0.5\nwbf_skip_box: 0.1\nyolo_iou: 0.95\n")
    result = load_config_file(path)
    assert result == {
        "wbf_iou": 0.5,
        "wbf_skip_box_thr": 0.1,
        "yolo_iou_threshold": 0.95,
    }


def test_unknown_key_raises_with_suggestion(tmp_path):
    path = _write(tmp_path, "wbf_iuo: 0.5\n")
    with pytest.raises(ConfigError) as exc:
        load_config_file(path)
    message = str(exc.value)
    assert "wbf_iuo" in message
    assert "wbf_iou" in message  # did-you-mean suggestion


def test_empty_file_returns_empty_dict(tmp_path):
    path = _write(tmp_path, "\n")
    assert load_config_file(path) == {}


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config_file(tmp_path / "does_not_exist.yaml")


def test_non_mapping_content_raises(tmp_path):
    path = _write(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ConfigError):
        load_config_file(path)


def test_bad_value_type_raises_naming_key(tmp_path):
    path = _write(tmp_path, "wbf_iou: not_a_number\n")
    with pytest.raises(ConfigError) as exc:
        load_config_file(path)
    assert "wbf_iou" in str(exc.value)


def test_full_key_parity_coercions(tmp_path):
    from pathlib import Path as _P

    text = (
        "dataset: /data/test\n"
        "output_dir: /tmp/out\n"
        "device: cuda:1\n"
        "batch_size: 16\n"
        "predict_threshold: 0.001\n"
        "weights: [1.0, 2, 3.0]\n"
        "skip_models: [yolo, deimv2]\n"
        "save_visualizations: false\n"
        "visualization_count: 4\n"
        "run_name: my_run\n"
        "rfdetr_weights: /w/rfdetr.pth\n"
        "yolo_weights: /w/yolo.pt\n"
        "deimv2_weights: /w/deimv2.pth\n"
        "deimv2_config: /w/deimv2.yml\n"
        "deimv2_dir: /w/deimv2\n"
        "log_level: debug\n"
    )
    result = load_config_file(_write(tmp_path, text))
    assert result["dataset_dir"] == _P("/data/test")
    assert result["output_dir"] == _P("/tmp/out")
    assert result["device"] == "cuda:1"
    assert result["batch_size"] == 16 and isinstance(result["batch_size"], int)
    assert result["predict_threshold"] == 0.001
    assert result["wbf_weights"] == (1.0, 2.0, 3.0)
    assert result["skip_models"] == ("yolo", "deimv2")
    assert result["save_visualizations"] is False
    assert result["visualization_count"] == 4
    assert result["run_name"] == "my_run"
    assert result["rfdetr_weights"] == _P("/w/rfdetr.pth")
    assert result["deimv2_dir"] == _P("/w/deimv2")
    assert result["log_level"] == "DEBUG"  # normalized upper


def test_resolve_precedence_cli_over_yaml_over_default():
    defaults = {
        "wbf_iou": 0.7,
        "wbf_skip_box_thr": 0.5,
        "batch_size": 8,
        "device": "cuda:0",
    }
    yaml_overrides = {"wbf_iou": 0.5, "wbf_skip_box_thr": 0.1}
    cli_overrides = {"wbf_iou": 0.6, "batch_size": 16}

    final, overrides = resolve(defaults, yaml_overrides, cli_overrides)

    assert final["wbf_iou"] == 0.6  # cli beats yaml
    assert final["wbf_skip_box_thr"] == 0.1  # yaml beats default
    assert final["batch_size"] == 16  # cli beats default
    assert final["device"] == "cuda:0"  # untouched default

    by_field = {o.field: o for o in overrides}
    assert set(by_field) == {"wbf_iou", "wbf_skip_box_thr", "batch_size"}
    assert by_field["wbf_iou"].source == "cli"
    assert by_field["wbf_skip_box_thr"].source == "yaml"
    assert by_field["batch_size"].source == "cli"
    assert by_field["wbf_skip_box_thr"].yaml_key == "wbf_skip_box"
    # Ordered by SPECS declaration order.
    assert [o.field for o in overrides] == [
        "batch_size",
        "wbf_iou",
        "wbf_skip_box_thr",
    ]


def test_resolve_no_overrides_is_empty_list():
    defaults = {"wbf_iou": 0.7}
    final, overrides = resolve(defaults, {}, {})
    assert final == {"wbf_iou": 0.7}
    assert overrides == []


def test_render_banner_lists_overrides_with_sources():
    overrides = [
        Override("wbf_iou", "wbf_iou", 0.5, "yaml"),
        Override("batch_size", "batch_size", 16, "cli"),
    ]
    banner = render_banner("configs/aggressive.yaml", overrides)

    assert "aggressive.yaml" in banner
    assert "2 overrides" in banner
    lines = banner.splitlines()
    iou_idx = next(i for i, l in enumerate(lines) if "wbf_iou" in l)
    bs_idx = next(i for i, l in enumerate(lines) if "batch_size" in l)
    assert iou_idx < bs_idx
    assert "[yaml]" in lines[iou_idx] and "0.5" in lines[iou_idx]
    assert "[cli]" in lines[bs_idx] and "16" in lines[bs_idx]


def test_render_banner_no_config_no_overrides():
    banner = render_banner(None, [])
    assert "0 overrides" in banner
    assert banner.strip()  # non-empty, no crash


def test_render_banner_formats_tuple_value():
    overrides = [Override("wbf_weights", "weights", (1.0, 2.0, 3.0), "yaml")]
    banner = render_banner("c.yaml", overrides)
    weights_line = next(l for l in banner.splitlines() if "weights" in l)
    assert "1" in weights_line and "2" in weights_line and "3" in weights_line
