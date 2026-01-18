from typing import List, Optional, Dict

import torch
import torch.nn as nn
import numpy as np

from lnp_ml.featurization.smiles import MPNNFeaturizer


class CachedMPNNEncoder(nn.Module):
    """
    带内存缓存的 D-MPNN 特征提取模块。

    - 使用预训练 chemprop 模型的 encoder 提取特征
    - 不可训练，不参与反向传播
    - 缓存已计算的 SMILES 特征，避免重复计算
    - forward 返回 Dict[str, Tensor]，key: "mpnn"
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        ensemble_paths: Optional[List[str]] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self._featurizer = MPNNFeaturizer(
            checkpoint_path=checkpoint_path,
            ensemble_paths=ensemble_paths,
            device=device,
        )
        self._cache: Dict[str, np.ndarray] = {}

    def forward(self, smiles_list: List[str]) -> Dict[str, torch.Tensor]:
        """
        SMILES 列表 -> Dict[str, Tensor]

        Returns:
            {"mpnn": (N, hidden_size)}
        """
        # 分离：已缓存 vs 需计算
        to_compute = [s for s in smiles_list if s not in self._cache]

        # 批量计算未缓存的
        if to_compute:
            new_features = self._featurizer.transform(to_compute)
            for idx, smiles in enumerate(to_compute):
                self._cache[smiles] = new_features["mpnn"][idx]

        # 按原顺序组装结果
        return {
            "mpnn": torch.from_numpy(np.stack([self._cache[s] for s in smiles_list]))
        }

    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """当前缓存的 SMILES 数量"""
        return len(self._cache)

    @property
    def hidden_size(self) -> int:
        """特征维度"""
        return self._featurizer.hidden_size
