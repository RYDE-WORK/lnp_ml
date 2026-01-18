import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict

from lnp_ml.featurization.smiles import RDKitFeaturizer


class CachedRDKitEncoder(nn.Module):
    """
    带内存缓存的 RDKit 特征提取模块。

    - 不可训练，不参与反向传播
    - 缓存已计算的 SMILES 特征，避免重复计算
    - forward 返回 Dict[str, Tensor]，keys: "morgan", "maccs", "desc"
    """

    def __init__(self, morgan_radius: int = 2, morgan_nbits: int = 1024) -> None:
        super().__init__()
        self._featurizer = RDKitFeaturizer(
            morgan_radius=morgan_radius,
            morgan_nbits=morgan_nbits,
        )
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}

    def forward(self, smiles_list: List[str]) -> Dict[str, torch.Tensor]:
        """
        SMILES 列表 -> Dict[str, Tensor]

        Returns:
            {"morgan": (N, 1024), "maccs": (N, 167), "desc": (N, 210)}
        """
        # 分离：已缓存 vs 需计算
        to_compute = [s for s in smiles_list if s not in self._cache]

        # 批量计算未缓存的
        if to_compute:
            new_features = self._featurizer.transform(to_compute)
            for idx, smiles in enumerate(to_compute):
                self._cache[smiles] = {
                    k: new_features[k][idx] for k in new_features
                }

        # 按原顺序组装结果
        keys = ["morgan", "maccs", "desc"]
        return {
            k: torch.from_numpy(np.stack([self._cache[s][k] for s in smiles_list]))
            for k in keys
        }

    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """当前缓存的 SMILES 数量"""
        return len(self._cache)
