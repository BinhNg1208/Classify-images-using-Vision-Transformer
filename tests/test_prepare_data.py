import json

import numpy as np
import pytest
from PIL import Image

from scripts.prepare_data import (
    check_counts, load_images, main, read_index, stratified_split,
)

WNIDS = ["n02", "n01", "n03"]  # cố ý không theo thứ tự để kiểm tra việc sắp xếp nhãn
PER_CLASS, VAL_PER_CLASS = 10, 2
EXPECTED = (3, 30, 6)


@pytest.fixture
def fake_root(tmp_path):
    """Thư mục giống Tiny ImageNet thu nhỏ: 3 lớp, 10 ảnh train/lớp, 2 ảnh val/lớp, ảnh 8x8."""
    root = tmp_path / "tiny"
    (root / "val" / "images").mkdir(parents=True)
    (root / "wnids.txt").write_text("\n".join(WNIDS) + "\n")
    (root / "words.txt").write_text(
        "n01\tgoldfish, Carassius auratus\nn02\tcat\nn03\tdog, domestic dog\nn99\tunused\n"
    )
    ann = []
    for c, w in enumerate(WNIDS):
        d = root / "train" / w / "images"
        d.mkdir(parents=True)
        for i in range(PER_CLASS):
            Image.new("RGB", (8, 8), (c * 80, i, 0)).save(d / f"{w}_{i}.JPEG")
        for i in range(VAL_PER_CLASS):
            name = f"val_{w}_{i}.JPEG"
            Image.new("RGB", (8, 8), (c * 80, 0, 0)).save(root / "val" / "images" / name)
            ann.append(f"{name}\t{w}\t0\t0\t8\t8")
    (root / "val" / "val_annotations.txt").write_text("\n".join(ann) + "\n")
    return root


def make_cfg(root, tmp_path):
    return {"seed": 42, "data": {"root": str(root), "img_size": 16, "val_ratio": 0.1,
                                 "cache_dir": str(tmp_path / "cache")}}


def test_read_index_sorts_classes_and_labels_val_from_annotations(fake_root):
    wnids, train, val = read_index(fake_root)
    assert wnids == ["n01", "n02", "n03"]
    assert len(train) == 30 and len(val) == 6
    assert all(p.parent.parent.name == wnids[y] for p, y in train)
    assert all(p.name.startswith(f"val_{wnids[y]}_") for p, y in val)


def test_check_counts_stops_with_clear_message_on_wrong_count(fake_root):
    wnids, train, val = read_index(fake_root)
    check_counts(wnids, train, val, EXPECTED)
    with pytest.raises(SystemExit, match="100000"):
        check_counts(wnids, train, val, (3, 100000, 6))


def test_stratified_split_is_disjoint_per_class_and_repeatable():
    labels = np.repeat(np.arange(3), 10)
    tr, va = stratified_split(labels, 0.1, seed=42)
    assert len(set(tr) & set(va)) == 0
    assert sorted(np.concatenate([tr, va])) == list(range(30))
    assert np.bincount(labels[va]).tolist() == [1, 1, 1]
    tr2, va2 = stratified_split(labels, 0.1, seed=42)
    assert tr.tolist() == tr2.tolist() and va.tolist() == va2.tolist()


def test_load_images_resizes_to_uint8_hwc(fake_root):
    _, train, _ = read_index(fake_root)
    x = load_images([p for p, _ in train[:4]], 16)
    assert x.shape == (4, 16, 16, 3) and x.dtype == np.uint8


def test_main_writes_cache_stats_and_is_stable_on_rerun(fake_root, tmp_path):
    cfg, out = make_cfg(fake_root, tmp_path), tmp_path / "outputs"
    main(cfg, out_dir=out, expected=EXPECTED)
    cache = tmp_path / "cache"
    shapes = {s: np.load(cache / f"{s}_x.npy").shape for s in ("train", "val", "test")}
    assert shapes == {"train": (27, 16, 16, 3), "val": (3, 16, 16, 3), "test": (6, 16, 16, 3)}
    assert json.loads((out / "results" / "class_names.json").read_text()) == ["goldfish", "cat", "dog"]
    stats = json.loads((out / "results" / "dataset_stats.json").read_text())
    assert stats["counts"] == {"train": 27, "val": 3, "test": 6}
    assert "\\" not in stats["cache"]["train"]["x"]  # đường dẫn dạng / để đọc được trên Linux
    assert (out / "figures" / "samples.png").exists()

    first = np.load(cache / "val_y.npy")
    main(cfg, out_dir=out, expected=EXPECTED)
    assert np.load(cache / "val_y.npy").tolist() == first.tolist()
    assert not list(cache.glob("*.tmp"))


def test_main_without_data_stops_with_download_instructions(tmp_path):
    cfg = make_cfg(tmp_path / "missing", tmp_path)
    with pytest.raises(SystemExit, match="cs231n.stanford.edu"):
        main(cfg, out_dir=tmp_path / "outputs", expected=EXPECTED)
