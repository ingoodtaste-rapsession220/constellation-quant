"""Phase 0 checkpoint tests — package imports, configs load, utils work.

These are the minimum that must pass before Phase 1 begins.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_top_level_package_imports():
    import constellation_quant
    assert constellation_quant.__version__ == "0.1.0"


def test_utils_public_api():
    from constellation_quant.utils import (
        Timer,
        get_device,
        get_logger,
        init_distributed,
        is_main_process,
        load_config,
        log_environment,
        merge_configs,
        set_seed,
        timed,
    )
    assert callable(get_logger)
    assert callable(load_config)
    assert callable(set_seed)


def test_subpackages_importable():
    for sub in [
        "data", "features", "graph",
        "models", "models.temporal", "models.graph_nn", "models.output_heads",
        "training", "ablation", "evaluation", "outputs", "utils",
    ]:
        __import__(f"constellation_quant.{sub}")


@pytest.mark.parametrize("name", [
    "data_config.yaml",
    "model_config.yaml",
    "training_config.yaml",
    "ablation_config.yaml",
    "feature_config.yaml",
    "paths.yaml",
])
def test_every_config_loads(name):
    from constellation_quant.utils import load_config
    cfg = load_config(CONFIGS / name)
    assert isinstance(cfg, dict)
    assert len(cfg) > 0


def test_env_expansion(tmp_path, monkeypatch):
    from constellation_quant.utils import load_config
    monkeypatch.setenv("DG_FAKE_PATH", "/fake/scratch")
    p = tmp_path / "c.yaml"
    p.write_text("root: ${DG_FAKE_PATH}/data\n")
    cfg = load_config(p)
    assert cfg["root"] == "/fake/scratch/data"


def test_config_inheritance(tmp_path):
    from constellation_quant.utils import load_config
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("a: 1\nb: {c: 2, d: 3}\n")
    child.write_text("extends: base.yaml\nb: {c: 20}\ne: 5\n")
    cfg = load_config(child)
    assert cfg == {"a": 1, "b": {"c": 20, "d": 3}, "e": 5}


def test_merge_configs_deep():
    from constellation_quant.utils import merge_configs
    base = {"x": {"y": 1, "z": 2}, "lst": [1, 2]}
    over = {"x": {"y": 10}, "lst": [9]}
    out = merge_configs(base, over)
    assert out == {"x": {"y": 10, "z": 2}, "lst": [9]}


def test_set_seed_is_deterministic():
    import random
    from constellation_quant.utils import set_seed
    set_seed(123, deterministic=False)
    a = random.random()
    set_seed(123, deterministic=False)
    b = random.random()
    assert a == b


def test_log_environment_has_keys():
    from constellation_quant.utils import log_environment
    env = log_environment()
    assert "python_version" in env
    assert "platform" in env


def test_timer_records_elapsed():
    import time
    from constellation_quant.utils import Timer
    with Timer("unit") as t:
        time.sleep(0.01)
    assert t.elapsed >= 0.01


def test_ablation_config_has_nine_variants():
    from constellation_quant.utils import load_config
    cfg = load_config(CONFIGS / "ablation_config.yaml")
    names = [v["name"] for v in cfg["variants"]]
    assert names == ["A", "B", "C", "D", "E", "F", "G", "H", "I"]


def test_slurm_templates_exist():
    slurm_dir = ROOT / "scripts" / "slurm"
    required = ["download.sh", "precompute_graphs.sh",
               "train_single.sh", "train_ddp.sh", "ablation_array.sh"]
    for name in required:
        path = slurm_dir / name
        assert path.exists(), f"missing SLURM template: {name}"
        assert os.access(path, os.X_OK), f"{name} not executable"
