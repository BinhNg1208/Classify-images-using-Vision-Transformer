"""Điểm tạo mô hình dùng chung; hợp đồng ở docs/05_model.md."""
from torch import nn

from .vit import VisionTransformer


def build_model(cfg: dict) -> nn.Module:
    """Tạo mô hình từ data/model trong config, không sửa config của bên gọi."""
    model_cfg = dict(cfg["model"])
    name = model_cfg.pop("name")
    if name == "vit":
        return VisionTransformer(
            img_size=cfg["data"]["img_size"],
            num_classes=cfg["data"]["num_classes"],
            **model_cfg,
        )
    # TV3 thêm nhánh resnet_small ở TASK-005.
    raise ValueError(f"model.name không hợp lệ: {name!r}. Tên hiện hỗ trợ: vit")
