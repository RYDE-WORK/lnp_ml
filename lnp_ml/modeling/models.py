"""LNP 多任务预测模型"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Literal

from lnp_ml.modeling.encoders import CachedRDKitEncoder, CachedMPNNEncoder
from lnp_ml.modeling.layers import TokenProjector, CrossModalAttention, FusionLayer
from lnp_ml.modeling.heads import MultiTaskHead


PoolingStrategy = Literal["concat", "avg", "max", "attention"]


# Token 维度配置（根据 ARCHITECTURE.md）
DEFAULT_INPUT_DIMS = {
    # Channel A: 化学特征
    "mpnn": 600,      # D-MPNN embedding
    "morgan": 1024,   # Morgan fingerprint
    "maccs": 167,     # MACCS keys
    "desc": 210,      # RDKit descriptors
    # Channel B: 配方/实验条件
    "comp": 5,        # 配方比例
    "phys": 12,       # 物理参数 one-hot
    "help": 4,        # Helper lipid one-hot
    "exp": 32,        # 实验条件 one-hot
}

# Token 顺序（前 4 个为 Channel A，后 4 个为 Channel B）
TOKEN_ORDER = ["mpnn", "morgan", "maccs", "desc", "comp", "phys", "help", "exp"]


class LNPModel(nn.Module):
    """
    LNP 药物递送性能预测模型。

    架构流程:
        1. Encoders: SMILES -> 化学特征; tabular -> 配方/实验特征
        2. TokenProjector: 统一到 d_model
        3. Stack: [B, 8, d_model]
        4. CrossModalAttention: Channel A (化学) <-> Channel B (配方/实验)
        5. FusionLayer: [B, 8, d_model] -> [B, fusion_dim]
        6. MultiTaskHead: 多任务预测
    """

    def __init__(
        self,
        # 模型维度
        d_model: int = 256,
        # Cross attention
        num_heads: int = 8,
        n_attn_layers: int = 4,
        # Fusion
        fusion_strategy: PoolingStrategy = "attention",
        # Head
        head_hidden_dim: int = 128,
        # Dropout
        dropout: float = 0.1,
        # MPNN encoder (可选，如果不用 MPNN 可以设为 None)
        mpnn_checkpoint: Optional[str] = None,
        mpnn_ensemble_paths: Optional[List[str]] = None,
        mpnn_device: str = "cpu",
        # 输入维度配置
        input_dims: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()

        self.input_dims = input_dims or DEFAULT_INPUT_DIMS
        self.d_model = d_model
        self.use_mpnn = mpnn_checkpoint is not None or mpnn_ensemble_paths is not None

        # ============ Encoders ============
        # RDKit encoder (always used)
        self.rdkit_encoder = CachedRDKitEncoder()

        # MPNN encoder (optional)
        if self.use_mpnn:
            self.mpnn_encoder = CachedMPNNEncoder(
                checkpoint_path=mpnn_checkpoint,
                ensemble_paths=mpnn_ensemble_paths,
                device=mpnn_device,
            )
        else:
            self.mpnn_encoder = None

        # ============ Token Projector ============
        # 根据是否使用 MPNN 调整输入维度
        proj_input_dims = {k: v for k, v in self.input_dims.items()}
        if not self.use_mpnn:
            proj_input_dims.pop("mpnn", None)

        self.token_projector = TokenProjector(
            input_dims=proj_input_dims,
            d_model=d_model,
            dropout=dropout,
        )

        # ============ Cross Modal Attention ============
        n_tokens = 8 if self.use_mpnn else 7
        split_idx = 4 if self.use_mpnn else 3  # Channel A 的 token 数量

        self.cross_attention = CrossModalAttention(
            d_model=d_model,
            num_heads=num_heads,
            n_layers=n_attn_layers,
            split_idx=split_idx,
            dropout=dropout,
        )

        # ============ Fusion Layer ============
        self.fusion = FusionLayer(
            d_model=d_model,
            n_tokens=n_tokens,
            strategy=fusion_strategy,
        )

        # ============ Multi-Task Head ============
        self.head = MultiTaskHead(
            in_dim=self.fusion.fusion_dim,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        smiles: List[str],
        tabular: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            smiles: SMILES 字符串列表，长度为 B
            tabular: Dict[str, Tensor]，包含:
                - "comp": [B, 5] 配方比例
                - "phys": [B, 12] 物理参数
                - "help": [B, 4] Helper lipid
                - "exp": [B, 32] 实验条件

        Returns:
            Dict[str, Tensor]:
                - "size": [B, 1]
                - "pdi": [B, 4]
                - "ee": [B, 3]
                - "delivery": [B, 1]
                - "biodist": [B, 7]
                - "toxic": [B, 2]
        """
        # 1. Encode SMILES
        rdkit_features = self.rdkit_encoder(smiles)  # {"morgan", "maccs", "desc"}

        # 2. 合并所有特征
        all_features: Dict[str, torch.Tensor] = {}

        # MPNN 特征（如果启用）
        if self.use_mpnn:
            mpnn_features = self.mpnn_encoder(smiles)
            all_features["mpnn"] = mpnn_features["mpnn"]

        # RDKit 特征
        all_features["morgan"] = rdkit_features["morgan"]
        all_features["maccs"] = rdkit_features["maccs"]
        all_features["desc"] = rdkit_features["desc"]

        # Tabular 特征
        all_features["comp"] = tabular["comp"]
        all_features["phys"] = tabular["phys"]
        all_features["help"] = tabular["help"]
        all_features["exp"] = tabular["exp"]

        # 3. Token Projector: 统一维度
        projected = self.token_projector(all_features)  # Dict[str, [B, d_model]]

        # 4. Stack tokens: [B, n_tokens, d_model]
        # 按顺序 stack：Channel A (化学) + Channel B (配方/实验)
        if self.use_mpnn:
            token_order = ["mpnn", "morgan", "maccs", "desc", "comp", "phys", "help", "exp"]
        else:
            token_order = ["morgan", "maccs", "desc", "comp", "phys", "help", "exp"]

        stacked = torch.stack([projected[k] for k in token_order], dim=1)

        # 5. Cross Modal Attention
        attended = self.cross_attention(stacked)

        # 6. Fusion
        fused = self.fusion(attended)

        # 7. Multi-Task Head
        outputs = self.head(fused)

        return outputs

    def clear_cache(self) -> None:
        """清空所有 encoder 的缓存"""
        self.rdkit_encoder.clear_cache()
        if self.mpnn_encoder is not None:
            self.mpnn_encoder.clear_cache()


class LNPModelWithoutMPNN(LNPModel):
    """不使用 MPNN 的简化版本"""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        n_attn_layers: int = 4,
        fusion_strategy: PoolingStrategy = "attention",
        head_hidden_dim: int = 128,
        dropout: float = 0.1,
        input_dims: Optional[Dict[str, int]] = None,
    ) -> None:
        # 移除 mpnn 维度
        dims = input_dims or DEFAULT_INPUT_DIMS.copy()
        dims.pop("mpnn", None)

        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            n_attn_layers=n_attn_layers,
            fusion_strategy=fusion_strategy,
            head_hidden_dim=head_hidden_dim,
            dropout=dropout,
            mpnn_checkpoint=None,
            mpnn_ensemble_paths=None,
            input_dims=dims,
        )

