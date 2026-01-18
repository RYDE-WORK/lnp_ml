import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class RegressionHead(nn.Module):
    """回归任务头：输出单个 float 值"""

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, in_dim] -> [B, 1]"""
        return self.net(x)


class ClassificationHead(nn.Module):
    """分类任务头：输出 logits"""

    def __init__(
        self, in_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, in_dim] -> [B, num_classes] (logits)"""
        return self.net(x)


class DistributionHead(nn.Module):
    """分布任务头：输出和为 1 的概率分布（用于 Biodistribution）"""

    def __init__(
        self, in_dim: int, num_outputs: int, hidden_dim: int = 128, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, in_dim] -> [B, num_outputs] (softmax, sum=1)"""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


class MultiTaskHead(nn.Module):
    """
    多任务预测头，根据任务配置自动创建对应的子头。

    输出:
        - size: [B, 1] 回归
        - pdi: [B, 4] 分类 logits
        - ee: [B, 3] 分类 logits
        - delivery: [B, 1] 回归
        - biodist: [B, 7] softmax 分布
        - toxic: [B, 2] 二分类 logits
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()

        # size: 回归 (log-transformed)
        self.size_head = RegressionHead(in_dim, hidden_dim, dropout)

        # PDI: 4 分类
        self.pdi_head = ClassificationHead(in_dim, num_classes=4, hidden_dim=hidden_dim, dropout=dropout)

        # Encapsulation Efficiency: 3 分类
        self.ee_head = ClassificationHead(in_dim, num_classes=3, hidden_dim=hidden_dim, dropout=dropout)

        # quantified_delivery: 回归 (z-scored)
        self.delivery_head = RegressionHead(in_dim, hidden_dim, dropout)

        # Biodistribution: 7 输出，softmax (sum=1)
        self.biodist_head = DistributionHead(in_dim, num_outputs=7, hidden_dim=hidden_dim, dropout=dropout)

        # toxic: 二分类
        self.toxic_head = ClassificationHead(in_dim, num_classes=2, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, in_dim] fusion 层输出

        Returns:
            Dict with keys:
                - "size": [B, 1]
                - "pdi": [B, 4] logits
                - "ee": [B, 3] logits
                - "delivery": [B, 1]
                - "biodist": [B, 7] probabilities (sum=1)
                - "toxic": [B, 2] logits
        """
        return {
            "size": self.size_head(x),
            "pdi": self.pdi_head(x),
            "ee": self.ee_head(x),
            "delivery": self.delivery_head(x),
            "biodist": self.biodist_head(x),
            "toxic": self.toxic_head(x),
        }
