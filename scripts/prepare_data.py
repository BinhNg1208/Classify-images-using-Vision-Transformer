"""Chuẩn bị Tiny ImageNet (TASK-002, FR-001).

Chia train gốc 90/10 phân tầng thành train/validation, dùng val gốc làm test,
xuất mảng cache uint8 (N, H, W, 3) và nhãn int64 vào `data.cache_dir`.

    python scripts/prepare_data.py                       # dữ liệu đã có ở data.root
    python scripts/prepare_data.py --download            # tải zip từ Stanford rồi chuẩn bị
"""
import argparse
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_config  # noqa: E402

URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
EXPECTED = (200, 100_000, 10_000)  # số lớp, ảnh train gốc, ảnh val gốc


def read_index(root):
    """Trả về (wnids đã sắp xếp, [(đường dẫn, nhãn)] train gốc, [(đường dẫn, nhãn)] val gốc)."""
    root = Path(root)
    wnids = sorted((root / "wnids.txt").read_text().split())
    label = {w: i for i, w in enumerate(wnids)}
    train = [(p, label[w]) for w in wnids for p in sorted((root / "train" / w / "images").glob("*.JPEG"))]
    val = []
    for line in (root / "val" / "val_annotations.txt").read_text().splitlines():
        if line.strip():
            name, w = line.split("\t")[:2]
            val.append((root / "val" / "images" / name, label[w]))
    return wnids, train, val


def check_counts(wnids, train, val, expected=EXPECTED):
    got = (len(wnids), len(train), len(val))
    if got != tuple(expected):
        sys.exit(f"Sai số lượng dữ liệu: có (lớp, train, val) = {got}, cần {tuple(expected)}. "
                 "File tải về có thể bị thiếu; xóa thư mục dữ liệu và tải lại.")


def stratified_split(labels, val_ratio, seed):
    """Chia chỉ số theo từng lớp; cùng seed cho cùng kết quả."""
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_val = round(len(idx) * val_ratio)
        va.append(idx[:n_val])
        tr.append(idx[n_val:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


def load_images(paths, img_size):
    x = np.empty((len(paths), img_size, img_size, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            im = im.convert("RGB")  # Tiny ImageNet có vài ảnh xám
            if im.size != (img_size, img_size):
                im = im.resize((img_size, img_size), Image.BICUBIC)
            x[i] = np.asarray(im)
    return x


def _save_npy(path, arr):
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, path)  # chạy lại hoặc bị ngắt giữa chừng không làm hỏng file cũ
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _class_names(root, wnids):
    words = dict(line.split("\t", 1) for line in (Path(root) / "words.txt").read_text().splitlines() if "\t" in line)
    return [words.get(w, w).split(",")[0].strip() for w in wnids]


def _download(root):
    root.parent.mkdir(parents=True, exist_ok=True)
    zip_path = root.parent / "tiny-imagenet-200.zip"
    if not zip_path.exists():
        print(f"Tải {URL} (khoảng 248 MB)...")
        urllib.request.urlretrieve(URL, zip_path.with_name(zip_path.name + ".part"))
        os.replace(zip_path.with_name(zip_path.name + ".part"), zip_path)
    print("Giải nén...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(root.parent)


def _save_samples(x, y, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(25, len(x))
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(x[i])
            ax.set_title(names[y[i]], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(cfg, out_dir="outputs", expected=EXPECTED, download=False):
    d = cfg["data"]
    root, cache, out = Path(d["root"]), Path(d["cache_dir"]), Path(out_dir)
    if not (root / "wnids.txt").exists():
        if not download:
            sys.exit(f"Không thấy dữ liệu ở '{root}'. Chạy lại với --download, hoặc tải tay {URL} "
                     f"và giải nén sao cho có '{root / 'wnids.txt'}'.")
        _download(root)

    wnids, train, val = read_index(root)
    check_counts(wnids, train, val, expected)
    print(f"{len(wnids)} lớp, {len(train)} ảnh train gốc, {len(val)} ảnh val gốc")

    labels = np.array([y for _, y in train], dtype=np.int64)
    tr_idx, va_idx = stratified_split(labels, d["val_ratio"], cfg["seed"])
    splits = {
        "train": [train[i] for i in tr_idx],
        "val": [train[i] for i in va_idx],
        "test": val,  # val gốc có nhãn làm test; test gốc không nhãn bị bỏ qua
    }

    cache.mkdir(parents=True, exist_ok=True)
    stats = {"dataset": "tiny_imagenet", "num_classes": len(wnids), "img_size": d["img_size"],
             "seed": cfg["seed"], "counts": {}, "cache": {}}
    for name, items in splits.items():
        print(f"Đọc {len(items)} ảnh {name}...")
        x = load_images([p for p, _ in items], d["img_size"])
        y = np.array([lab for _, lab in items], dtype=np.int64)
        _save_npy(cache / f"{name}_x.npy", x)
        _save_npy(cache / f"{name}_y.npy", y)
        stats["counts"][name] = len(items)
        stats["cache"][name] = {"x": (cache / f"{name}_x.npy").as_posix(), "y": (cache / f"{name}_y.npy").as_posix(),
                                "shape": list(x.shape), "bytes": int(x.nbytes)}
        if name == "train":  # train xếp theo lớp, lấy ngẫu nhiên để lưới mẫu có nhiều lớp
            pick = np.random.default_rng(cfg["seed"]).choice(len(x), min(25, len(x)), replace=False)
            sample_x, sample_y = x[pick], y[pick]
    # Train và validation lấy từ hai tập chỉ số không giao nhau của cùng train gốc.
    assert not set(map(str, (p for p, _ in splits["train"]))) & set(map(str, (p for p, _ in splits["val"])))

    names = _class_names(root, wnids)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "results" / "class_names.json").write_text(json.dumps(names, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "results" / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    _save_samples(sample_x, sample_y, names, out / "figures" / "samples.png")
    print(json.dumps(stats["counts"]), f"cache {sum(c['bytes'] for c in stats['cache'].values()) / 1e9:.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/vit_small.yaml")
    ap.add_argument("--download", action="store_true", help=f"tải {URL} nếu chưa có dữ liệu")
    args = ap.parse_args()
    main(load_config(args.config), download=args.download)
