"""Điểm tạo mô hình dùng chung; hợp đồng ở docs/05_model.md."""
from torch import nn

from .cnn import ResNetSmall, match_param_count
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

    if name == "resnet_small":
        return ResNetSmall(
            num_classes=cfg["data"]["num_classes"],
            **{
                key: tuple(value) if isinstance(value, list) else value
                for key, value in model_cfg.items()
            },
        )

    raise ValueError(
        f"model.name không hợp lệ: {name!r}. "
        "Tên hiện hỗ trợ: vit, resnet_small"
    )
