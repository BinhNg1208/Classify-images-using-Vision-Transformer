"""Test cho src/utils.py (TASK-001): AC-013 và phần ghi file an toàn của AC-008."""
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.utils import atomic_save, count_params, load_config, make_run_name, set_seed

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

VALID_YAML = """
run_name: vit_small
seed: 42
data:
  name: tiny_imagenet
  root: data/tiny-imagenet-200
  img_size: 64
  num_classes: 200
  batch_size: 64
  num_workers: 4
  val_ratio: 0.1
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
model:
  name: vit
train:
  epochs: 50
  optimizer: adamw
  lr: 2.5e-4
  weight_decay: 0.05
"""


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(text, encoding="utf-8")
    return p


# ---------- load_config (AC-013) ----------

def test_load_config_returns_dict_for_valid_file(tmp_path):
    cfg = load_config(write(tmp_path, VALID_YAML))
    assert cfg["data"]["img_size"] == 64
    assert cfg["train"]["lr"] == pytest.approx(2.5e-4)


def test_load_config_names_every_missing_key(tmp_path):
    text = VALID_YAML.replace("  mean: [0.485, 0.456, 0.406]\n", "").replace("  lr: 2.5e-4\n", "")
    with pytest.raises(ValueError) as e:
        load_config(write(tmp_path, text))
    assert "data.mean" in str(e.value)
    assert "train.lr" in str(e.value)


def test_load_config_reports_keys_of_missing_section(tmp_path):
    text = VALID_YAML.replace("model:\n  name: vit\n", "")
    with pytest.raises(ValueError, match=r"model\.name"):
        load_config(write(tmp_path, text))


def test_load_config_rejects_empty_file(tmp_path):
    with pytest.raises(ValueError, match="run_name"):
        load_config(write(tmp_path, ""))


@pytest.mark.parametrize("name", ["vit_small.yaml", "cnn_baseline.yaml"])
def test_shipped_configs_are_valid(name):
    load_config(CONFIGS / name)


def test_vit_and_cnn_share_data_and_train_settings():
    # So sánh công bằng: hai mô hình chỉ được khác nhau ở phần model.
    vit = load_config(CONFIGS / "vit_small.yaml")
    cnn = load_config(CONFIGS / "cnn_baseline.yaml")
    assert vit["data"] == cnn["data"]
    assert vit["train"] == cnn["train"]
    assert vit["seed"] == cnn["seed"]


# ---------- make_run_name ----------

def test_make_run_name_appends_seed():
    assert make_run_name({"run_name": "vit_small", "seed": 42}) == "vit_small_seed42"


# ---------- set_seed ----------

def test_set_seed_makes_random_sources_repeatable():
    set_seed(0)
    first = (random.random(), np.random.rand(), torch.rand(3))
    set_seed(0)
    second = (random.random(), np.random.rand(), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


# ---------- count_params ----------

def test_count_params_counts_weights_and_biases():
    assert count_params(nn.Linear(3, 2)) == 3 * 2 + 2


# ---------- atomic_save (DEC-005) ----------

def test_atomic_save_writes_file_loadable_with_weights_only(tmp_path):
    p = tmp_path / "ckpt" / "last.pt"
    atomic_save({"model": {"w": torch.ones(2)}, "epoch": 3, "scaler": None}, p)
    loaded = torch.load(p, weights_only=True)
    assert loaded["epoch"] == 3
    assert torch.equal(loaded["model"]["w"], torch.ones(2))


def test_atomic_save_keeps_old_file_when_save_fails(tmp_path):
    p = tmp_path / "last.pt"
    atomic_save({"epoch": 1}, p)
    with pytest.raises(Exception):
        atomic_save({"bad": lambda x: x}, p)  # lambda không pickle được
    assert torch.load(p, weights_only=True) == {"epoch": 1}
    assert [f.name for f in tmp_path.iterdir()] == ["last.pt"]
