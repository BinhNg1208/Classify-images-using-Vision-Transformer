"""ResNet nhỏ làm baseline cho ViT (TASK-005)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.utils import count_params

class BasicBlock(nn.Module):
    """BasicBlock chuẩn với BatchNorm và residual connection."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class ResNetSmall(nn.Module):
    """ResNet nhỏ cho ảnh 64x64."""

    def __init__(
        self,
        num_classes: int = 200,
        widths: tuple[int, int, int, int] = (64, 128, 256, 512),
        blocks: tuple[int, int, int, int] = (1, 1, 1, 1),
        stem_stride: int = 2,
    ) -> None:
        super().__init__()

        if len(widths) != 4:
            raise ValueError("widths phải có đúng 4 phần tử")
        if len(blocks) != 4:
            raise ValueError("blocks phải có đúng 4 phần tử")
        if any(w <= 0 for w in widths):
            raise ValueError("widths phải là số nguyên dương")
        if any(b <= 0 for b in blocks):
            raise ValueError("blocks phải là số nguyên dương")
        if stem_stride not in (1, 2):
            raise ValueError("stem_stride phải là 1 hoặc 2")

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                widths[0],
                kernel_size=3,
                stride=stem_stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(
            widths[0],
            widths[0],
            blocks[0],
            stride=1,
        )
        self.layer2 = self._make_layer(
            widths[0],
            widths[1],
            blocks[1],
            stride=2,
        )
        self.layer3 = self._make_layer(
            widths[1],
            widths[2],
            blocks[2],
            stride=2,
        )
        self.layer4 = self._make_layer(
            widths[2],
            widths[3],
            blocks[3],
            stride=2,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[3], num_classes)

        self._init_weights()

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [
            BasicBlock(
                in_channels,
                out_channels,
                stride=stride,
            )
        ]

        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                f"input phải có shape (B, 3, H, W), nhận {tuple(x.shape)}"
            )

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        return self.fc(x)


@dataclass(frozen=True)
class ResNetConfig:
    widths: tuple[int, int, int, int]
    blocks: tuple[int, int, int, int]
    stem_stride: int

def match_param_count(
    target_params: int,
    num_classes: int = 200,
    max_relative_error: float = 0.05,
) -> tuple[ResNetConfig, int]:
    """
    Tìm cấu hình ResNet nhỏ có số tham số gần target nhất.

    Chỉ trả về cấu hình khi sai lệch không vượt quá max_relative_error.
    """

    if target_params <= 0:
        raise ValueError("target_params phải dương")

    if not 0 < max_relative_error < 1:
        raise ValueError("max_relative_error phải thuộc (0, 1)")

    # Ưu tiên kiến trúc nhỏ, downsample sớm theo DEC-007.
    candidates: list[ResNetConfig] = []

    widths_options = (
        (32, 64, 128, 256),
        (40, 80, 160, 320),
        (48, 96, 192, 384),
        (56, 112, 224, 448),
        (64, 128, 256, 512),
    )

    block_options = (
        (1, 1, 1, 1),
        (1, 2, 2, 1),
        (1, 2, 2, 2),
        (1, 2, 3, 2),
        (1, 3, 3, 2),
        (2, 2, 2, 2),
        (2, 3, 3, 2),
        (2, 3, 4, 2),
        (2, 3, 4, 3),
        (2, 4, 4, 2),
    )

    for widths in widths_options:
        for blocks in block_options:
            for stem_stride in (2, 1):
                config = ResNetConfig(
                    widths=widths,
                    blocks=blocks,
                    stem_stride=stem_stride,
                )
                candidates.append(config)

    best: tuple[float, ResNetConfig, int] | None = None

    for config in candidates:
        with torch.device("meta"):
            model = ResNetSmall(
                num_classes=num_classes,
                widths=config.widths,
                blocks=config.blocks,
                stem_stride=config.stem_stride,
            )

        params = count_params(model)
        relative_error = abs(params - target_params) / target_params

        candidate = (relative_error, config, params)

        if best is None or candidate[0] < best[0]:
            best = candidate

    assert best is not None

    relative_error, config, params = best

    if relative_error > max_relative_error:
        raise ValueError(
            "Không tìm được cấu hình ResNet trong giới hạn "
            f"{max_relative_error:.1%}: "
            f"target={target_params:,}, best={params:,}, "
            f"error={relative_error:.2%}"
        )

    return config, params