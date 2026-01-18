import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Literal, Union


PoolingStrategy = Literal["concat", "avg", "max", "attention"]


class FusionLayer(nn.Module):
    """
    将多个 token 融合成单个向量。

    输入: Dict[str, Tensor] 或 [B, n_tokens, d_model]
    输出: [B, fusion_dim]

    策略:
        - concat: [B, n_tokens, d_model] -> [B, n_tokens * d_model]
        - avg: [B, n_tokens, d_model] -> [B, d_model]
        - max: [B, n_tokens, d_model] -> [B, d_model]
        - attention: [B, n_tokens, d_model] -> [B, d_model] (learnable attention pooling)
    """

    def __init__(
        self,
        d_model: int,
        n_tokens: int,
        strategy: PoolingStrategy = "attention",
    ) -> None:
        """
        Args:
            d_model: 每个 token 的维度
            n_tokens: token 数量（如 8）
            strategy: 融合策略
        """
        super().__init__()
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.strategy = strategy

        if strategy == "concat":
            self.fusion_dim = n_tokens * d_model
        else:
            self.fusion_dim = d_model

        # Attention pooling: learnable query
        if strategy == "attention":
            self.attn_query = nn.Parameter(torch.randn(1, 1, d_model))
            self.attn_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: Dict[str, Tensor] 每个 [B, d_model]，或已 stack 的 [B, n_tokens, d_model]

        Returns:
            [B, fusion_dim]
        """
        # 如果输入是 dict，先 stack
        if isinstance(x, dict):
            x = torch.stack(list(x.values()), dim=1)  # [B, n_tokens, d_model]

        if self.strategy == "concat":
            return x.flatten(start_dim=1)  # [B, n_tokens * d_model]

        elif self.strategy == "avg":
            return x.mean(dim=1)  # [B, d_model]

        elif self.strategy == "max":
            return x.max(dim=1).values  # [B, d_model]

        elif self.strategy == "attention":
            return self._attention_pooling(x)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _attention_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Attention pooling: 用可学习 query 对 tokens 做加权求和

        Args:
            x: [B, n_tokens, d_model]

        Returns:
            [B, d_model]
        """
        B = x.size(0)
        # query: [1, 1, d_model] -> [B, 1, d_model]
        query = self.attn_query.expand(B, -1, -1)

        # Attention scores: [B, 1, n_tokens]
        keys = self.attn_proj(x)  # [B, n_tokens, d_model]
        scores = torch.bmm(query, keys.transpose(1, 2)) / (self.d_model ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # [B, 1, n_tokens]

        # Weighted sum: [B, 1, d_model] -> [B, d_model]
        out = torch.bmm(attn_weights, x).squeeze(1)
        return out
