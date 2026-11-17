import pickle

import numpy as np
import pytest
import torch

from src.data import CachedDataset, build_loader

N, S = 20, 16
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


@pytest.fixture
def cfg(tmp_path):
    """Cache giả cùng định dạng prepare_data.py: {split}_x.npy uint8 NHWC, {split}_y.npy int64."""
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        np.save(tmp_path / f"{split}_x.npy", rng.integers(0, 256, (N, S, S, 3), dtype=np.uint8))
        np.save(tmp_path / f"{split}_y.npy", np.arange(N, dtype=np.int64) % 4)
    return {"data": {"cache_dir": str(tmp_path), "img_size": S, "batch_size": 8, "num_workers": 0,
                     "mean": MEAN, "std": STD, "randaugment": True, "random_erasing": 0.0}}


def test_train_loader_gives_normalized_float_batches(cfg):
    x, y = next(iter(build_loader(cfg, "train")))
    assert x.shape == (8, 3, S, S) and x.dtype == torch.float32
    assert y.shape == (8,) and y.dtype == torch.int64


def test_val_loader_only_normalizes_in_order(cfg, tmp_path):
    loader = build_loader(cfg, "val")
    x, y = next(iter(loader))
    raw = np.load(tmp_path / "val_x.npy")[:8]
    expected = (torch.from_numpy(raw).permute(0, 3, 1, 2).float() / 255
                - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
    assert torch.allclose(x, expected, atol=1e-5)
    assert y.tolist() == (np.arange(8) % 4).tolist()
    assert len(loader.dataset) == N  # val/test không bỏ ảnh nào


def test_train_transform_augments_but_keeps_shape(cfg):
    train = build_loader(cfg, "train").dataset
    val = build_loader(cfg, "val").dataset
    torch.manual_seed(0)
    a = torch.stack([train[i][0] for i in range(N)])
    b = torch.stack([CachedDataset(train.x_path, train.y, val.transform)[i][0] for i in range(N)])
    assert a.shape == b.shape == (N, 3, S, S)
    assert not torch.allclose(a, b)


def test_missing_cache_tells_user_to_run_prepare_data(cfg, tmp_path):
    cfg["data"]["cache_dir"] = str(tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        build_loader(cfg, "train")


def test_dataset_pickles_without_copying_the_array(cfg):
    # Windows tạo worker bằng spawn: dataset bị pickle sang từng worker, không được kéo theo cả mảng 1,1 GB.
    ds = build_loader(cfg, "train").dataset
    ds[0]  # đã mở mmap trong tiến trình chính
    assert len(pickle.dumps(ds)) < 10_000


def test_loader_options_follow_config(cfg):
    train, val = build_loader(cfg, "train"), build_loader(cfg, "val")
    assert train.drop_last and not val.drop_last
    assert train.pin_memory == torch.cuda.is_available()  # chỉ có ích khi có GPU
    assert not train.persistent_workers  # num_workers = 0


def test_loader_works_with_worker_processes(cfg):
    cfg["data"]["num_workers"] = 2
    loader = build_loader(cfg, "train")
    assert loader.persistent_workers
    assert sum(len(y) for _, y in loader) == 16  # 20 ảnh, batch 8, bỏ batch lẻ
