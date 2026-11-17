"""ViT tự cài: Pre-LN, CLS token, SDPA và attention trước dropout (TASK-004)."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DropPath(nn.Module):
    """Bỏ cả nhánh residual theo từng ảnh, giữ kỳ vọng bằng cách chia keep_prob."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_prob phải thuộc [0, 1)")
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class PatchEmbedding(nn.Module):
    """Conv stride bằng patch_size: ảnh RGB -> chuỗi (B, số patch, embed_dim)."""

    def __init__(self, img_size: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1:] != (3, self.img_size, self.img_size):
            raise ValueError(
                f"input phải có shape (B, 3, {self.img_size}, {self.img_size}), "
                f"nhận {tuple(x.shape)}"
            )
        return self.proj(x).flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    """Hai nhánh dùng chung qkv/proj; chỉ nhánh tường minh trả attention weights."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, return_attn: bool = False) -> tuple[Tensor, Tensor | None]:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = None
        if return_attn:
            attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            out = self.attn_drop(attn) @ v  # trả attn gốc, trước dropout
        else:
            # SDPA vẫn áp dropout ở eval nếu truyền p > 0, nên phải tắt tường minh.
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        out = out.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(out)), attn


class TransformerEncoderBlock(nn.Module):
    """Pre-LayerNorm và hai residual có stochastic depth."""

    def __init__(
        self, embed_dim: int, num_heads: int, mlp_ratio: float,
        dropout: float, drop_path_rate: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim), nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: Tensor, return_attn: bool = False) -> tuple[Tensor, Tensor | None]:
        out, attn = self.attn(self.norm1(x), return_attn=return_attn)
        x = x + self.drop_path(out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class VisionTransformer(nn.Module):
    """ViT cho ảnh vuông; cấu hình lấy qua build_model(cfg), không dùng pretrained."""

    def __init__(
        self, img_size: int = 64, patch_size: int = 8, num_classes: int = 200,
        embed_dim: int = 256, depth: int = 8, num_heads: int = 8,
        mlp_ratio: float = 4.0, dropout: float = 0.0, drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        for name, value in (
            ("img_size", img_size), ("patch_size", patch_size),
            ("num_classes", num_classes), ("embed_dim", embed_dim),
            ("depth", depth), ("num_heads", num_heads),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} phải là số nguyên dương")
        if img_size % patch_size:
            raise ValueError("img_size phải chia hết cho patch_size")
        if embed_dim % num_heads:
            raise ValueError("embed_dim phải chia hết cho num_heads")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or int(embed_dim * mlp_ratio) < 1:
            raise ValueError("mlp_ratio phải dương và tạo ít nhất một hidden unit")
        for name, value in (("dropout", dropout), ("drop_path_rate", drop_path_rate)):
            if not 0 <= value < 1:
                raise ValueError(f"{name} phải thuộc [0, 1)")

        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        rates = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout, rate)
            for rate in rates
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, return_attn: bool = False) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Trả logits; return_attn=True thêm một attention trước dropout cho mỗi layer."""
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.pos_drop(torch.cat((cls, x), dim=1) + self.pos_embed)
        attns = []
        for block in self.blocks:
            x, attn = block(x, return_attn=return_attn)
            if attn is not None:
                attns.append(attn)
        logits = self.head(self.norm(x)[:, 0])
        return (logits, attns) if return_attn else logits
