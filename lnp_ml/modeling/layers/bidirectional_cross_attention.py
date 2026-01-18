import torch
import torch.nn as nn
from typing import Tuple


class CrossAttentionLayer(nn.Module):
    """单层双向交叉注意力"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # A -> B: A as Q, B as K/V
        self.cross_attn_a2b = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # B -> A: B as Q, A as K/V
        self.cross_attn_b2a = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # LayerNorm + FFN for channel A
        self.norm_a1 = nn.LayerNorm(d_model)
        self.norm_a2 = nn.LayerNorm(d_model)
        self.ffn_a = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # LayerNorm + FFN for channel B
        self.norm_b1 = nn.LayerNorm(d_model)
        self.norm_b2 = nn.LayerNorm(d_model)
        self.ffn_b = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            a: [B, seq_len, d_model]
            b: [B, seq_len, d_model]

        Returns:
            (a_out, b_out): 更新后的两个 channel
        """
        # Cross attention: A attends to B
        a_attn, _ = self.cross_attn_a2b(query=a, key=b, value=b)
        a = self.norm_a1(a + a_attn)
        a = self.norm_a2(a + self.ffn_a(a))

        # Cross attention: B attends to A
        b_attn, _ = self.cross_attn_b2a(query=b, key=a, value=a)
        b = self.norm_b1(b + b_attn)
        b = self.norm_b2(b + self.ffn_b(b))

        return a, b


class CrossModalAttention(nn.Module):
    """
    双向交叉注意力模块。

    输入 stacked tokens [B, 8, d_model]，split 成两个 channel 后执行
    n_layers 层双向交叉注意力。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        n_layers: int,
        split_idx: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            d_model: 特征维度
            num_heads: 注意力头数，d_head = d_model / num_heads
            n_layers: 交叉注意力层数
            split_idx: channel split 的位置，默认 4 (0:4, 4:)
            dropout: dropout 比例
        """
        super().__init__()
        self.split_idx = split_idx

        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 8, d_model] stacked tokens

        Returns:
            [B, 8, d_model] 融合后的 tokens
        """
        # Split: [B, 8, d_model] -> [B, 4, d_model], [B, 4, d_model]
        a = x[:, : self.split_idx, :]
        b = x[:, self.split_idx :, :]

        # N layers of bidirectional cross attention
        for layer in self.layers:
            a, b = layer(a, b)

        # Concat back: [B, 8, d_model]
        return torch.cat([a, b], dim=1)

