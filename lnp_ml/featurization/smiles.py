"""SMILES 分子特征提取器"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np

# Suppress RDKit deprecation warnings
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors

import torch
from chemprop.utils import load_checkpoint
from chemprop.features import mol2graph


@dataclass
class RDKitFeaturizer:
    """
    SMILES -> RDKit 特征向量，返回 Dict[str, np.ndarray]。
    """

    morgan_radius: int = 2
    morgan_nbits: int = 1024

    def transform(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """SMILES 特征字典 -> value: (N, D_i) arrays"""
        encoded = [self._encode_one(s) for s in smiles_list]
        return {
            "morgan": np.vstack([enc["morgan"] for enc in encoded]),
            "maccs": np.vstack([enc["maccs"] for enc in encoded]),
            "desc": np.vstack([enc["desc"] for enc in encoded])
        }

    def _encode_morgan(self, mol: Chem.Mol) -> np.ndarray:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=self.morgan_radius, nBits=self.morgan_nbits
        ).ToList(), dtype=np.float32)

    def _encode_maccs(self, mol: Chem.Mol) -> np.ndarray:
        return np.array(MACCSkeys.GenMACCSKeys(mol).ToList(), dtype=np.float32)
    
    def _encode_desc(self, mol: Chem.Mol) -> np.ndarray:
        # 使用 float64 计算，然后 clip 到 float32 范围，避免 overflow
        desc_values = list(Descriptors.CalcMolDescriptors(mol).values())
        arr = np.array(desc_values, dtype=np.float64)
        # 替换 inf/nan，clip 到 float32 范围
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
        arr = np.clip(arr, -1e10, 1e10)
        return arr.astype(np.float32)

    def _encode_one(self, smiles: str) -> Dict[str, np.ndarray]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")
        
        return {
            "morgan": self._encode_morgan(mol),
            "maccs": self._encode_maccs(mol),
            "desc": self._encode_desc(mol)
        }


@dataclass
class MPNNFeaturizer:
    """
    SMILES -> D-MPNN 预训练特征向量 (N, hidden_size=600)。
    
    从训练好的 chemprop 模型中提取 D-MPNN 编码器的输出作为分子特征。
    
    Args:
        checkpoint_path: 模型检查点路径（.pt文件）
        device: 计算设备 ("cpu" 或 "cuda")
        ensemble_paths: 可选，多个模型路径列表用于集成（取平均）
    """
    checkpoint_path: Optional[str] = None
    device: str = "cpu"
    ensemble_paths: Optional[List[str]] = None
    
    # 内部状态（不由用户设置）
    _encoders: List = field(default_factory=list, init=False, repr=False)
    _hidden_size: int = field(default=0, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """延迟初始化，在首次调用 transform 时加载模型"""
        if self.checkpoint_path is None and self.ensemble_paths is None:
            raise ValueError("必须提供 checkpoint_path 或 ensemble_paths")
        
    def _lazy_init(self) -> None:
        """延迟加载模型，避免在创建对象时就加载大模型"""
        if self._initialized:
            return
            
        device = torch.device(self.device)
        paths = self.ensemble_paths or [self.checkpoint_path]
        
        for path in paths:
            model = load_checkpoint(path, device=device)
            model.eval()
            # 提取 MPNEncoder（D-MPNN 核心部分）
            encoder = model.encoder.encoder[0]
            # 冻结参数
            for param in encoder.parameters():
                param.requires_grad = False
            self._encoders.append(encoder)
            self._hidden_size = encoder.hidden_size
        
        self._initialized = True
    
    def transform(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """
        SMILES 列表 -> tuple of (N, hidden_size) array
        
        Args:
            smiles_list: SMILES 字符串列表
            
        Returns:
            tuple 包含一个形状为 (N, hidden_size) 的 numpy 数组
            如果使用集成模型，返回所有模型输出的平均值
        """
        self._lazy_init()
        
        # 验证 SMILES 有效性
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smi!r}")
        
        # 构建分子图（批量处理）
        batch_mol_graph = mol2graph(smiles_list)
        
        # 从所有编码器提取特征
        all_features = []
        with torch.no_grad():
            for encoder in self._encoders:
                features = encoder(batch_mol_graph)
                all_features.append(features.cpu().numpy())
        
        # 如果是集成模型，取平均
        if len(all_features) > 1:
            features_array = np.mean(all_features, axis=0).astype(np.float32)
        else:
            features_array = all_features[0].astype(np.float32)
        
        return {
            "mpnn": features_array
        }
    
    @property
    def hidden_size(self) -> int:
        """返回特征维度"""
        self._lazy_init()
        return self._hidden_size
    
    @property
    def n_models(self) -> int:
        """返回集成模型数量"""
        self._lazy_init()
        return len(self._encoders)
