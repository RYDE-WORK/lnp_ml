import torch
import torch.nn as nn
from typing import Dict


class TokenProjector(nn.Module):
    """
    将不同维度的特征投影到统一的 d_model 维度。

    每个特征分支的流程：
    [B, input_dim_i] -> BN -> Linear -> [B, d_model] -> ReLU -> BN -> Dropout -> * sigmoid(weight_i)
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            input_dims: 各特征的输入维度，如 {"morgan": 1024, "maccs": 167, "desc": 210}
            d_model: 统一的输出维度
            dropout: dropout 比例
        """
        super().__init__()
        self.keys = list(input_dims.keys())

        # 为每个特征分支创建投影层
        self.projectors = nn.ModuleDict()
        for key, in_dim in input_dims.items():
            self.projectors[key] = nn.Sequential(
                nn.BatchNorm1d(in_dim),
                nn.Linear(in_dim, d_model),
                nn.ReLU(),
                nn.BatchNorm1d(d_model),
                nn.Dropout(dropout),
            )

        # 每个分支的可学习权重（初始化为 0，sigmoid 后为 0.5）
        self.weights = nn.ParameterDict({
            key: nn.Parameter(torch.zeros(1)) for key in self.keys
        })

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Dict[str, Tensor]，每个 tensor 形状为 (B, input_dim_i)

        Returns:
            Dict[str, Tensor]，每个 tensor 形状为 (B, d_model)
        """
        out = {}
        for key in self.keys:
            x = self.projectors[key](features[key])
            w = torch.sigmoid(self.weights[key])
            out[key] = x * w
        return out

