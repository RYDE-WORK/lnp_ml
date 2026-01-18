"""数据集处理模块"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ============ 列名配置 ============

# SMILES 列
SMILES_COL = "smiles"

# comp token: 配方比例 [5]
COMP_COLS = [
    "Cationic_Lipid_to_mRNA_weight_ratio",
    "Cationic_Lipid_Mol_Ratio",
    "Phospholipid_Mol_Ratio",
    "Cholesterol_Mol_Ratio",
    "PEG_Lipid_Mol_Ratio",
]

# phys token: 物理/实验参数 one-hot [12]
# 需要从原始列生成 one-hot
PHYS_ONEHOT_SPECS = {
    "Purity": ["Pure", "Crude"],
    "Mix_type": ["Microfluidic", "Pipetting"],
    "Cargo_type": ["mRNA", "pDNA", "siRNA"],
    "Target_or_delivered_gene": ["FFL", "Peptide_barcode", "hEPO", "FVII", "GFP"],
}

# help token: Helper lipid one-hot [4]
HELP_COLS = [
    "Helper_lipid_ID_DOPE",
    "Helper_lipid_ID_DOTAP",
    "Helper_lipid_ID_DSPC",
    "Helper_lipid_ID_MDOA",
]

# exp token: 实验条件 one-hot [32]
EXP_ONEHOT_SPECS = {
    "Model_type": ["A549", "BDMC", "BMDM", "HBEC_ALI", "HEK293T", "HeLa", "IGROV1", "Mouse", "RAW264p7"],
    "Delivery_target": ["body", "dendritic_cell", "generic_cell", "liver", "lung", "lung_epithelium", "macrophage", "muscle", "spleen"],
    "Route_of_administration": ["in_vitro", "intramuscular", "intratracheal", "intravenous"],
    "Batch_or_individual_or_barcoded": ["Barcoded", "Individual"],
    "Value_name": ["log_luminescence", "luminescence", "FFL_silencing", "Peptide_abundance", "hEPO", "FVII_silencing", "GFP_delivery", "Discretized_luminescence"],
}

# Target 列
TARGET_REGRESSION = ["size", "quantified_delivery"]
TARGET_CLASSIFICATION_PDI = ["PDI_0_0to0_2", "PDI_0_2to0_3", "PDI_0_3to0_4", "PDI_0_4to0_5"]
TARGET_CLASSIFICATION_EE = ["Encapsulation_Efficiency_EE<50", "Encapsulation_Efficiency_50<=EE<80", "Encapsulation_Efficiency_80<EE<=100"]
TARGET_TOXIC = "toxic"
TARGET_BIODIST = [
    "Biodistribution_lymph_nodes",
    "Biodistribution_heart",
    "Biodistribution_liver",
    "Biodistribution_spleen",
    "Biodistribution_lung",
    "Biodistribution_kidney",
    "Biodistribution_muscle",
]


def get_onehot_cols(prefix: str, values: List[str]) -> List[str]:
    """生成 one-hot 列名"""
    return [f"{prefix}_{v}" for v in values]


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理原始 DataFrame，生成模型所需的所有列。
    
    Args:
        df: 原始 DataFrame
        
    Returns:
        处理后的 DataFrame，包含所有需要的列
    """
    df = df.copy()
    
    # 1. 处理 phys token 的 one-hot 列（如果不存在则生成）
    for col, values in PHYS_ONEHOT_SPECS.items():
        for v in values:
            onehot_col = f"{col}_{v}"
            if onehot_col not in df.columns:
                if col in df.columns:
                    df[onehot_col] = (df[col] == v).astype(float)
                else:
                    df[onehot_col] = 0.0
    
    # 2. 处理 exp token 的 one-hot 列（如果不存在则生成）
    for col, values in EXP_ONEHOT_SPECS.items():
        for v in values:
            onehot_col = f"{col}_{v}"
            if onehot_col not in df.columns:
                if col in df.columns:
                    df[onehot_col] = (df[col] == v).astype(float)
                else:
                    df[onehot_col] = 0.0
    
    # 3. 确保 comp 列存在且为 float
    for col in COMP_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    
    # 4. 确保 help 列存在
    for col in HELP_COLS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0).astype(float)
    
    # 5. 处理 target 列
    # size: 已经 log 过，填充缺失值
    if "size" in df.columns:
        df["size"] = pd.to_numeric(df["size"], errors="coerce")
    
    # quantified_delivery: 已经 z-score 过
    if "quantified_delivery" in df.columns:
        df["quantified_delivery"] = pd.to_numeric(df["quantified_delivery"], errors="coerce")
    
    # toxic: 0/1
    if TARGET_TOXIC in df.columns:
        df[TARGET_TOXIC] = pd.to_numeric(df[TARGET_TOXIC], errors="coerce").fillna(-1).astype(int)
    
    # PDI 和 EE 的 one-hot 分类
    for col in TARGET_CLASSIFICATION_PDI + TARGET_CLASSIFICATION_EE:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(float)
    
    # Biodistribution
    for col in TARGET_BIODIST:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    
    return df


def get_phys_cols() -> List[str]:
    """获取 phys token 的所有列名"""
    cols = []
    for col, values in PHYS_ONEHOT_SPECS.items():
        cols.extend(get_onehot_cols(col, values))
    return cols


def get_exp_cols() -> List[str]:
    """获取 exp token 的所有列名"""
    cols = []
    for col, values in EXP_ONEHOT_SPECS.items():
        cols.extend(get_onehot_cols(col, values))
    return cols


@dataclass
class LNPDatasetConfig:
    """数据集配置"""
    comp_cols: List[str] = None
    phys_cols: List[str] = None
    help_cols: List[str] = None
    exp_cols: List[str] = None
    
    def __post_init__(self):
        self.comp_cols = self.comp_cols or COMP_COLS
        self.phys_cols = self.phys_cols or get_phys_cols()
        self.help_cols = self.help_cols or HELP_COLS
        self.exp_cols = self.exp_cols or get_exp_cols()


class LNPDataset(Dataset):
    """
    LNP 数据集，用于 PyTorch DataLoader。
    
    返回:
        - smiles: str
        - tabular: Dict[str, Tensor] with keys "comp", "phys", "help", "exp"
        - targets: Dict[str, Tensor] with keys "size", "pdi", "ee", "delivery", "biodist", "toxic"
        - mask: Dict[str, Tensor] 标记哪些 target 有效（非缺失）
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[LNPDatasetConfig] = None,
    ):
        self.config = config or LNPDatasetConfig()
        self.df = process_dataframe(df)
        
        # 提取数据
        self.smiles = self.df[SMILES_COL].tolist()
        
        # Tabular features
        self.comp = self.df[self.config.comp_cols].values.astype(np.float32)
        self.phys = self.df[self.config.phys_cols].values.astype(np.float32)
        self.help = self.df[self.config.help_cols].values.astype(np.float32)
        self.exp = self.df[self.config.exp_cols].values.astype(np.float32)
        
        # Targets
        self.size = self.df["size"].values.astype(np.float32) if "size" in self.df.columns else None
        self.delivery = self.df["quantified_delivery"].values.astype(np.float32) if "quantified_delivery" in self.df.columns else None
        self.toxic = self.df[TARGET_TOXIC].values.astype(np.int64) if TARGET_TOXIC in self.df.columns else None
        
        # PDI: one-hot -> class index
        if all(col in self.df.columns for col in TARGET_CLASSIFICATION_PDI):
            pdi_onehot = self.df[TARGET_CLASSIFICATION_PDI].values
            self.pdi = np.argmax(pdi_onehot, axis=1).astype(np.int64)
            self.pdi_valid = pdi_onehot.sum(axis=1) > 0
        else:
            self.pdi = None
            self.pdi_valid = None
        
        # EE: one-hot -> class index
        if all(col in self.df.columns for col in TARGET_CLASSIFICATION_EE):
            ee_onehot = self.df[TARGET_CLASSIFICATION_EE].values
            self.ee = np.argmax(ee_onehot, axis=1).astype(np.int64)
            self.ee_valid = ee_onehot.sum(axis=1) > 0
        else:
            self.ee = None
            self.ee_valid = None
        
        # Biodistribution
        if all(col in self.df.columns for col in TARGET_BIODIST):
            self.biodist = self.df[TARGET_BIODIST].values.astype(np.float32)
            self.biodist_valid = self.biodist.sum(axis=1) > 0
        else:
            self.biodist = None
            self.biodist_valid = None
    
    def __len__(self) -> int:
        return len(self.smiles)
    
    def __getitem__(self, idx: int) -> Dict:
        item = {
            "smiles": self.smiles[idx],
            "tabular": {
                "comp": torch.from_numpy(self.comp[idx]),
                "phys": torch.from_numpy(self.phys[idx]),
                "help": torch.from_numpy(self.help[idx]),
                "exp": torch.from_numpy(self.exp[idx]),
            },
            "targets": {},
            "mask": {},
        }
        
        # Targets and masks
        if self.size is not None:
            item["targets"]["size"] = torch.tensor(self.size[idx], dtype=torch.float32)
            item["mask"]["size"] = torch.tensor(not np.isnan(self.size[idx]), dtype=torch.bool)
        
        if self.delivery is not None:
            item["targets"]["delivery"] = torch.tensor(self.delivery[idx], dtype=torch.float32)
            item["mask"]["delivery"] = torch.tensor(not np.isnan(self.delivery[idx]), dtype=torch.bool)
        
        if self.toxic is not None:
            item["targets"]["toxic"] = torch.tensor(self.toxic[idx], dtype=torch.long)
            item["mask"]["toxic"] = torch.tensor(self.toxic[idx] >= 0, dtype=torch.bool)
        
        if self.pdi is not None:
            item["targets"]["pdi"] = torch.tensor(self.pdi[idx], dtype=torch.long)
            item["mask"]["pdi"] = torch.tensor(self.pdi_valid[idx], dtype=torch.bool)
        
        if self.ee is not None:
            item["targets"]["ee"] = torch.tensor(self.ee[idx], dtype=torch.long)
            item["mask"]["ee"] = torch.tensor(self.ee_valid[idx], dtype=torch.bool)
        
        if self.biodist is not None:
            item["targets"]["biodist"] = torch.from_numpy(self.biodist[idx])
            item["mask"]["biodist"] = torch.tensor(self.biodist_valid[idx], dtype=torch.bool)
        
        return item


def collate_fn(batch: List[Dict]) -> Dict:
    """
    自定义 collate 函数，用于 DataLoader。
    
    Returns:
        - smiles: List[str]
        - tabular: Dict[str, Tensor] with batched tensors
        - targets: Dict[str, Tensor] with batched tensors
        - mask: Dict[str, Tensor] with batched masks
    """
    smiles = [item["smiles"] for item in batch]
    
    tabular = {
        key: torch.stack([item["tabular"][key] for item in batch])
        for key in batch[0]["tabular"].keys()
    }
    
    targets = {}
    mask = {}
    for key in batch[0]["targets"].keys():
        targets[key] = torch.stack([item["targets"][key] for item in batch])
        mask[key] = torch.stack([item["mask"][key] for item in batch])
    
    return {
        "smiles": smiles,
        "tabular": tabular,
        "targets": targets,
        "mask": mask,
    }


def load_dataset(
    path: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[LNPDataset, LNPDataset, LNPDataset]:
    """
    加载并划分数据集。
    
    Args:
        path: CSV 文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
        
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    df = pd.read_csv(path)
    
    # 随机打乱
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    
    config = LNPDatasetConfig()
    
    return (
        LNPDataset(train_df, config),
        LNPDataset(val_df, config),
        LNPDataset(test_df, config),
    )
