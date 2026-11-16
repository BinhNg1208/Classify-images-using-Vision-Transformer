"""Tiện ích dùng chung cho cả nhóm: nạp config, seed, đếm tham số, ghi file an toàn.

Hợp đồng config ở docs/02_architecture.md; ghi file an toàn theo DEC-005.
"""
from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

REQUIRED_KEYS: tuple[str, ...] = (
    "run_name",
    "seed",
    "data.name",
    "data.root",
    "data.img_size",
    "data.num_classes",
    "data.batch_size",
    "data.num_workers",
    "data.val_ratio",
    "data.mean",
    "data.std",
    "model.name",
    "train.epochs",
    "train.optimizer",
    "train.lr",
    "train.weight_decay",
)


def _has_key(cfg: dict, dotted: str) -> bool:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def load_config(path: str | Path) -> dict:
    """Đọc file YAML và kiểm tra đủ khóa bắt buộc; thiếu thì báo tất cả khóa thiếu một lần."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: config phải là một mapping YAML")
    missing = [k for k in REQUIRED_KEYS if not _has_key(cfg, k)]
    if missing:
        raise ValueError(f"{path}: thiếu khóa bắt buộc: {', '.join(missing)}")
    return cfg


def make_run_name(cfg: dict) -> str:
    """Tên lần chạy `<run_name>_seed<seed>`, để các seed không ghi đè log/checkpoint của nhau."""
    return f"{cfg['run_name']}_seed{cfg['seed']}"


def set_seed(seed: int) -> None:
    """Cố định seed cho random, numpy và torch (kể cả CUDA nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # gọi luôn cả seed của mọi GPU


def count_params(model: nn.Module) -> int:
    """Tổng số tham số của mô hình, dùng để so ViT với CNN."""
    return sum(p.numel() for p in model.parameters())


def atomic_save(obj: Any, path: str | Path) -> None:
    """torch.save ra file tạm cùng thư mục rồi đổi tên, để bị ngắt giữa chừng không hỏng file cũ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(obj, f)
        os.replace(tmp, path)  # đổi tên nguyên tử khi cùng ổ đĩa
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
