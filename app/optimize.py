"""
配方优化模拟程序

通过迭代式 Grid Search 寻找最优 LNP 配方，最大化目标器官的 Biodistribution。

使用方法:
    python -m app.optimize --smiles "CC(C)..." --organ liver
"""

import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm
import typer

from lnp_ml.config import MODELS_DIR
from lnp_ml.dataset import (
    LNPDataset,
    LNPDatasetConfig,
    collate_fn,
    SMILES_COL,
    COMP_COLS,
    HELP_COLS,
    TARGET_BIODIST,
    get_phys_cols,
    get_exp_cols,
)
from lnp_ml.modeling.predict import load_model

app = typer.Typer()

# ============ 参数配置 ============

# 可用的目标器官
AVAILABLE_ORGANS = ["lymph_nodes", "heart", "liver", "spleen", "lung", "kidney", "muscle"]

# comp token 参数范围
COMP_PARAM_RANGES = {
    "Cationic_Lipid_to_mRNA_weight_ratio": (0.05, 0.30),
    "Cationic_Lipid_Mol_Ratio": (0.05, 0.80),
    "Phospholipid_Mol_Ratio": (0.00, 0.80),
    "Cholesterol_Mol_Ratio": (0.00, 0.80),
    "PEG_Lipid_Mol_Ratio": (0.00, 0.05),
}

# 最小 step size
MIN_STEP_SIZE = 0.01

# 迭代策略：每个迭代的 step_size
ITERATION_STEP_SIZES = [0.10, 0.02, 0.01]

# Helper lipid 选项
HELPER_LIPID_OPTIONS = ["DOPE", "DSPC", "DOTAP"]

# Route of administration 选项
ROUTE_OPTIONS = ["intravenous", "intramuscular"]


@dataclass
class Formulation:
    """配方数据结构"""
    # comp token
    cationic_lipid_to_mrna_ratio: float
    cationic_lipid_mol_ratio: float
    phospholipid_mol_ratio: float
    cholesterol_mol_ratio: float
    peg_lipid_mol_ratio: float
    # 离散选项
    helper_lipid: str = "DOPE"
    route: str = "intravenous"
    # 预测结果（填充后）
    biodist_predictions: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "Cationic_Lipid_to_mRNA_weight_ratio": self.cationic_lipid_to_mrna_ratio,
            "Cationic_Lipid_Mol_Ratio": self.cationic_lipid_mol_ratio,
            "Phospholipid_Mol_Ratio": self.phospholipid_mol_ratio,
            "Cholesterol_Mol_Ratio": self.cholesterol_mol_ratio,
            "PEG_Lipid_Mol_Ratio": self.peg_lipid_mol_ratio,
            "helper_lipid": self.helper_lipid,
            "route": self.route,
        }
    
    def get_biodist(self, organ: str) -> float:
        """获取指定器官的 biodistribution 预测值"""
        col = f"Biodistribution_{organ}"
        return self.biodist_predictions.get(col, 0.0)


def generate_grid_values(
    center: float,
    step_size: float,
    min_val: float,
    max_val: float,
    radius: int = 2,
) -> List[float]:
    """
    围绕中心点生成网格值。
    
    Args:
        center: 中心值
        step_size: 步长
        min_val: 最小值
        max_val: 最大值
        radius: 扩展半径（生成 2*radius+1 个点）
    
    Returns:
        网格值列表
    """
    values = []
    for i in range(-radius, radius + 1):
        val = center + i * step_size
        if min_val <= val <= max_val:
            values.append(round(val, 4))
    return sorted(set(values))


def generate_initial_grid(step_size: float) -> List[Tuple[float, float, float, float, float]]:
    """
    生成初始搜索网格（满足 mol ratio 和为 1 的约束）。
    
    Returns:
        List of (cationic_ratio, cationic_mol, phospholipid_mol, cholesterol_mol, peg_mol)
    """
    grid = []
    
    # Cationic_Lipid_to_mRNA_weight_ratio
    weight_ratios = np.arange(0.05, 0.31, step_size)
    
    # PEG: 单独处理，范围很小
    peg_values = np.arange(0.00, 0.06, MIN_STEP_SIZE)  # PEG 始终用 0.01 步长
    
    # 其他三个 mol ratio 需要满足和为 1 - PEG
    mol_step = step_size
    
    for weight_ratio in weight_ratios:
        for peg in peg_values:
            remaining = 1.0 - peg
            # 生成满足约束的组合
            for cationic_mol in np.arange(0.05, min(0.81, remaining + 0.001), mol_step):
                for phospholipid_mol in np.arange(0.00, min(0.81, remaining - cationic_mol + 0.001), mol_step):
                    cholesterol_mol = remaining - cationic_mol - phospholipid_mol
                    # 检查约束
                    if 0.00 <= cholesterol_mol <= 0.80:
                        grid.append((
                            round(weight_ratio, 4),
                            round(cationic_mol, 4),
                            round(phospholipid_mol, 4),
                            round(cholesterol_mol, 4),
                            round(peg, 4),
                        ))
    
    return grid


def generate_refined_grid(
    seeds: List[Formulation],
    step_size: float,
    radius: int = 2,
) -> List[Tuple[float, float, float, float, float]]:
    """
    围绕种子点生成精细化网格。
    
    Args:
        seeds: 种子配方列表
        step_size: 步长
        radius: 扩展半径
    
    Returns:
        新的网格点列表
    """
    grid_set = set()
    
    for seed in seeds:
        # Weight ratio
        weight_ratios = generate_grid_values(
            seed.cationic_lipid_to_mrna_ratio, step_size, 0.05, 0.30, radius
        )
        
        # PEG (始终用最小步长)
        peg_values = generate_grid_values(
            seed.peg_lipid_mol_ratio, MIN_STEP_SIZE, 0.00, 0.05, radius
        )
        
        # Mol ratios
        cationic_mols = generate_grid_values(
            seed.cationic_lipid_mol_ratio, step_size, 0.05, 0.80, radius
        )
        phospholipid_mols = generate_grid_values(
            seed.phospholipid_mol_ratio, step_size, 0.00, 0.80, radius
        )
        
        for weight_ratio in weight_ratios:
            for peg in peg_values:
                remaining = 1.0 - peg
                for cationic_mol in cationic_mols:
                    for phospholipid_mol in phospholipid_mols:
                        cholesterol_mol = remaining - cationic_mol - phospholipid_mol
                        # 检查约束
                        if (0.05 <= cationic_mol <= 0.80 and
                            0.00 <= phospholipid_mol <= 0.80 and
                            0.00 <= cholesterol_mol <= 0.80 and
                            0.00 <= peg <= 0.05):
                            grid_set.add((
                                round(weight_ratio, 4),
                                round(cationic_mol, 4),
                                round(phospholipid_mol, 4),
                                round(cholesterol_mol, 4),
                                round(peg, 4),
                            ))
    
    return list(grid_set)


def create_dataframe_from_formulations(
    smiles: str,
    grid: List[Tuple[float, float, float, float, float]],
    helper_lipids: List[str],
    routes: List[str],
) -> pd.DataFrame:
    """
    从配方网格创建 DataFrame。
    
    使用固定的 phys token（Pure, Microfluidic, mRNA, FFL）和 exp token（Mouse, body, luminescence）。
    """
    rows = []
    
    for comp_values in grid:
        for helper in helper_lipids:
            for route in routes:
                row = {
                    SMILES_COL: smiles,
                    # comp token
                    "Cationic_Lipid_to_mRNA_weight_ratio": comp_values[0],
                    "Cationic_Lipid_Mol_Ratio": comp_values[1],
                    "Phospholipid_Mol_Ratio": comp_values[2],
                    "Cholesterol_Mol_Ratio": comp_values[3],
                    "PEG_Lipid_Mol_Ratio": comp_values[4],
                    # phys token (固定值)
                    "Purity_Pure": 1.0,
                    "Purity_Crude": 0.0,
                    "Mix_type_Microfluidic": 1.0,
                    "Mix_type_Pipetting": 0.0,
                    "Cargo_type_mRNA": 1.0,
                    "Cargo_type_pDNA": 0.0,
                    "Cargo_type_siRNA": 0.0,
                    "Target_or_delivered_gene_FFL": 1.0,
                    "Target_or_delivered_gene_Peptide_barcode": 0.0,
                    "Target_or_delivered_gene_hEPO": 0.0,
                    "Target_or_delivered_gene_FVII": 0.0,
                    "Target_or_delivered_gene_GFP": 0.0,
                    # help token
                    "Helper_lipid_ID_DOPE": 1.0 if helper == "DOPE" else 0.0,
                    "Helper_lipid_ID_DOTAP": 1.0 if helper == "DOTAP" else 0.0,
                    "Helper_lipid_ID_DSPC": 1.0 if helper == "DSPC" else 0.0,
                    "Helper_lipid_ID_MDOA": 0.0,  # 不使用
                    # exp token (固定值)
                    "Model_type_Mouse": 1.0,
                    "Delivery_target_body": 1.0,
                    f"Route_of_administration_{route}": 1.0,
                    "Batch_or_individual_or_barcoded_Individual": 1.0,
                    "Value_name_luminescence": 1.0,
                    # 存储配方元信息
                    "_helper_lipid": helper,
                    "_route": route,
                }
                # 其他 exp token 默认为 0
                for col in get_exp_cols():
                    if col not in row:
                        row[col] = 0.0
                
                rows.append(row)
    
    return pd.DataFrame(rows)


def predict_biodist(
    model: torch.nn.Module,
    df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    使用模型预测 biodistribution。
    
    Returns:
        添加了预测列的 DataFrame
    """
    dataset = LNPDataset(df)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    all_biodist_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            smiles = batch["smiles"]
            tabular = {k: v.to(device) for k, v in batch["tabular"].items()}
            
            outputs = model(smiles, tabular)
            
            # biodist 输出是 softmax 后的概率分布 [B, 7]
            biodist_pred = outputs["biodist"].cpu().numpy()
            all_biodist_preds.append(biodist_pred)
    
    biodist_preds = np.concatenate(all_biodist_preds, axis=0)
    
    # 添加到 DataFrame
    for i, col in enumerate(TARGET_BIODIST):
        df[f"pred_{col}"] = biodist_preds[:, i]
    
    return df


def select_top_k(
    df: pd.DataFrame,
    organ: str,
    k: int = 20,
) -> List[Formulation]:
    """
    选择 top-k 配方。
    
    Args:
        df: 包含预测结果的 DataFrame
        organ: 目标器官
        k: 选择数量
    
    Returns:
        Top-k 配方列表
    """
    pred_col = f"pred_Biodistribution_{organ}"
    if pred_col not in df.columns:
        raise ValueError(f"Prediction column {pred_col} not found")
    
    # 排序并去重
    df_sorted = df.sort_values(pred_col, ascending=False)
    
    # 创建配方对象
    formulations = []
    seen = set()
    
    for _, row in df_sorted.iterrows():
        key = (
            row["Cationic_Lipid_to_mRNA_weight_ratio"],
            row["Cationic_Lipid_Mol_Ratio"],
            row["Phospholipid_Mol_Ratio"],
            row["Cholesterol_Mol_Ratio"],
            row["PEG_Lipid_Mol_Ratio"],
            row["_helper_lipid"],
            row["_route"],
        )
        
        if key not in seen:
            seen.add(key)
            formulation = Formulation(
                cationic_lipid_to_mrna_ratio=row["Cationic_Lipid_to_mRNA_weight_ratio"],
                cationic_lipid_mol_ratio=row["Cationic_Lipid_Mol_Ratio"],
                phospholipid_mol_ratio=row["Phospholipid_Mol_Ratio"],
                cholesterol_mol_ratio=row["Cholesterol_Mol_Ratio"],
                peg_lipid_mol_ratio=row["PEG_Lipid_Mol_Ratio"],
                helper_lipid=row["_helper_lipid"],
                route=row["_route"],
                biodist_predictions={
                    col: row[f"pred_{col}"] for col in TARGET_BIODIST
                },
            )
            formulations.append(formulation)
            
            if len(formulations) >= k:
                break
    
    return formulations


def optimize(
    smiles: str,
    organ: str,
    model: torch.nn.Module,
    device: torch.device,
    top_k: int = 20,
    batch_size: int = 256,
) -> List[Formulation]:
    """
    执行配方优化。
    
    Args:
        smiles: SMILES 字符串
        organ: 目标器官
        model: 训练好的模型
        device: 计算设备
        top_k: 每轮保留的最优配方数
        batch_size: 预测批次大小
    
    Returns:
        最终 top-k 配方列表
    """
    logger.info(f"Starting optimization for organ: {organ}")
    logger.info(f"SMILES: {smiles}")
    
    seeds = None
    
    for iteration, step_size in enumerate(ITERATION_STEP_SIZES):
        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {iteration + 1}/{len(ITERATION_STEP_SIZES)}, step_size={step_size}")
        logger.info(f"{'='*60}")
        
        # 生成网格
        if seeds is None:
            # 第一次迭代：生成完整初始网格
            logger.info("Generating initial grid...")
            grid = generate_initial_grid(step_size)
        else:
            # 后续迭代：围绕种子点精细化
            logger.info(f"Generating refined grid around {len(seeds)} seeds...")
            grid = generate_refined_grid(seeds, step_size, radius=2)
        
        logger.info(f"Grid size: {len(grid)} comp combinations")
        
        # 扩展到所有 helper lipid 和 route 组合
        total_combinations = len(grid) * len(HELPER_LIPID_OPTIONS) * len(ROUTE_OPTIONS)
        logger.info(f"Total combinations: {total_combinations}")
        
        # 创建 DataFrame
        df = create_dataframe_from_formulations(
            smiles, grid, HELPER_LIPID_OPTIONS, ROUTE_OPTIONS
        )
        
        # 预测
        logger.info("Running predictions...")
        df = predict_biodist(model, df, device, batch_size)
        
        # 选择 top-k
        seeds = select_top_k(df, organ, top_k)
        
        # 显示当前最优
        best = seeds[0]
        logger.info(f"Current best Biodistribution_{organ}: {best.get_biodist(organ):.4f}")
        logger.info(f"Best formulation: {best.to_dict()}")
    
    return seeds


def format_results(formulations: List[Formulation], organ: str) -> pd.DataFrame:
    """格式化结果为 DataFrame"""
    rows = []
    for i, f in enumerate(formulations):
        row = {
            "rank": i + 1,
            f"Biodistribution_{organ}": f.get_biodist(organ),
            **f.to_dict(),
        }
        # 添加其他器官的预测
        for col in TARGET_BIODIST:
            if col != f"Biodistribution_{organ}":
                row[col] = f.biodist_predictions.get(col, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


@app.command()
def main(
    smiles: str = typer.Option(..., "--smiles", "-s", help="Cationic lipid SMILES string"),
    organ: str = typer.Option(..., "--organ", "-o", help=f"Target organ: {AVAILABLE_ORGANS}"),
    model_path: Path = typer.Option(
        MODELS_DIR / "final" / "model.pt",
        "--model", "-m",
        help="Path to trained model checkpoint"
    ),
    output_path: Optional[Path] = typer.Option(
        None, "--output", "-O",
        help="Output CSV path (optional)"
    ),
    top_k: int = typer.Option(20, "--top-k", "-k", help="Number of top formulations to return"),
    batch_size: int = typer.Option(256, "--batch-size", "-b", help="Prediction batch size"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", "--device", "-d", help="Device"),
):
    """
    配方优化程序：寻找最大化目标器官 Biodistribution 的最优 LNP 配方。
    
    示例:
        python -m app.optimize --smiles "CC(C)..." --organ liver
        python -m app.optimize -s "CC(C)..." -o spleen -k 10
    """
    # 验证器官
    if organ not in AVAILABLE_ORGANS:
        logger.error(f"Invalid organ: {organ}. Available: {AVAILABLE_ORGANS}")
        raise typer.Exit(1)
    
    # 加载模型
    logger.info(f"Loading model from {model_path}")
    device = torch.device(device)
    model = load_model(model_path, device)
    
    # 执行优化
    results = optimize(
        smiles=smiles,
        organ=organ,
        model=model,
        device=device,
        top_k=top_k,
        batch_size=batch_size,
    )
    
    # 格式化并显示结果
    df_results = format_results(results, organ)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TOP {top_k} FORMULATIONS FOR {organ.upper()}")
    logger.info(f"{'='*60}")
    print(df_results.to_string(index=False))
    
    # 保存结果
    if output_path:
        df_results.to_csv(output_path, index=False)
        logger.success(f"Results saved to {output_path}")
    
    return df_results


if __name__ == "__main__":
    app()

