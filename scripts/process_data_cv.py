"""内部数据 Cross-Validation 划分脚本：支持随机划分或基于 Amine 的分组划分"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import typer
from loguru import logger

from lnp_ml.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from lnp_ml.dataset import (
    process_dataframe,
    SMILES_COL,
    COMP_COLS,
    HELP_COLS,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION_PDI,
    TARGET_CLASSIFICATION_EE,
    TARGET_TOXIC,
    TARGET_BIODIST,
    get_phys_cols,
    get_exp_cols,
)

app = typer.Typer()


def random_cv_split(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> List[dict]:
    """
    随机 shuffle 进行 Cross-Validation 划分。
    
    步骤：
    1. 打乱所有样本
    2. 将样本分成 n_folds 个容器
    3. 对于每个 fold i：
       - validation = container[i]
       - test = container[(i+1) % n_folds]
       - train = 其余所有
    
    Args:
        df: 输入 DataFrame
        n_folds: 折数
        seed: 随机种子
        
    Returns:
        List of dicts，每个 dict 包含 train_df, val_df, test_df
    """
    # 打乱所有样本
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_samples = len(df_shuffled)
    
    logger.info(f"Total {n_samples} samples for random CV split")
    
    # 将样本分成 n_folds 个容器
    indices = np.arange(n_samples)
    containers = np.array_split(indices, n_folds)
    
    # 打印每个容器的大小
    for i, container in enumerate(containers):
        logger.info(f"  Container {i}: {len(container)} samples")
    
    # 生成每个 fold 的数据
    fold_splits = []
    for i in range(n_folds):
        val_indices = containers[i]
        test_indices = containers[(i + 1) % n_folds]
        train_indices = np.concatenate([
            containers[j] for j in range(n_folds) 
            if j != i and j != (i + 1) % n_folds
        ])
        
        train_df = df_shuffled.iloc[train_indices].reset_index(drop=True)
        val_df = df_shuffled.iloc[val_indices].reset_index(drop=True)
        test_df = df_shuffled.iloc[test_indices].reset_index(drop=True)
        
        fold_splits.append({
            "train": train_df,
            "val": val_df,
            "test": test_df,
        })
        
        logger.info(
            f"Fold {i}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
    
    return fold_splits


def amine_based_cv_split(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    amine_col: str = "Amine",
) -> List[dict]:
    """
    基于 Amine 列进行 Cross-Validation 划分。
    
    步骤：
    1. 按 amine_col 分组
    2. 打乱分组顺序
    3. 将分组 round-robin 分配到 n_folds 个容器
    4. 对于每个 fold i：
       - validation = container[i]
       - test = container[(i+1) % n_folds]
       - train = 其余所有
    
    Args:
        df: 输入 DataFrame
        n_folds: 折数
        seed: 随机种子
        amine_col: 用于分组的列名
        
    Returns:
        List of dicts，每个 dict 包含 train_df, val_df, test_df
    """
    # 获取唯一的 amine 并打乱
    unique_amines = df[amine_col].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_amines)
    
    logger.info(f"Found {len(unique_amines)} unique amines")
    
    # Round-robin 分配到 n_folds 个容器
    containers = [[] for _ in range(n_folds)]
    for i, amine in enumerate(unique_amines):
        containers[i % n_folds].append(amine)
    
    # 打印每个容器的大小
    for i, container in enumerate(containers):
        container_samples = df[df[amine_col].isin(container)]
        logger.info(f"  Container {i}: {len(container)} amines, {len(container_samples)} samples")
    
    # 生成每个 fold 的数据
    fold_splits = []
    for i in range(n_folds):
        val_amines = set(containers[i])
        test_amines = set(containers[(i + 1) % n_folds])
        train_amines = set()
        for j in range(n_folds):
            if j != i and j != (i + 1) % n_folds:
                train_amines.update(containers[j])
        
        train_df = df[df[amine_col].isin(train_amines)].reset_index(drop=True)
        val_df = df[df[amine_col].isin(val_amines)].reset_index(drop=True)
        test_df = df[df[amine_col].isin(test_amines)].reset_index(drop=True)
        
        fold_splits.append({
            "train": train_df,
            "val": val_df,
            "test": test_df,
        })
        
        logger.info(
            f"Fold {i}: train={len(train_df)} ({len(train_amines)} amines), "
            f"val={len(val_df)} ({len(val_amines)} amines), "
            f"test={len(test_df)} ({len(test_amines)} amines)"
        )
    
    return fold_splits


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "internal_corrected.csv",
    output_dir: Path = PROCESSED_DATA_DIR / "cv",
    n_folds: int = 5,
    seed: int = 42,
    amine_col: str = "Amine",
    scaffold_split: bool = typer.Option(
        False,
        "--scaffold-split",
        help="使用基于 Amine 的 scaffold splitting（默认：随机 shuffle）",
    ),
):
    """
    Cross-Validation 数据划分。
    
    支持两种划分方式：
    - 随机划分（默认）：直接 shuffle 所有样本
    - Scaffold splitting（--scaffold-split）：基于 Amine 分组，确保同一 Amine 的数据在同一组
    
    划分比例约为 train:val:test ≈ 3:1:1
    
    输出结构:
        - processed/cv/fold_0/train.parquet
        - processed/cv/fold_0/val.parquet  
        - processed/cv/fold_0/test.parquet
        - processed/cv/fold_1/...
        - processed/cv/feature_columns.txt
    """
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # 处理数据（列对齐、one-hot 生成等）
    logger.info("Processing dataframe...")
    df = process_dataframe(df)
    
    # 如果使用 scaffold split，检查 amine 列是否存在
    if scaffold_split:
        # 重新加载原始数据获取 Amine 列（process_dataframe 可能不会保留它）
        original_df = pd.read_csv(input_path)
        if amine_col not in original_df.columns:
            logger.error(f"Column '{amine_col}' not found in data. Available columns: {list(original_df.columns)}")
            raise typer.Exit(1)
        if amine_col not in df.columns:
            df[amine_col] = original_df[amine_col].values
    
    # 定义要保留的列
    phys_cols = get_phys_cols()
    exp_cols = get_exp_cols()
    
    keep_cols = (
        [SMILES_COL]
        + COMP_COLS
        + phys_cols
        + HELP_COLS
        + exp_cols
        + TARGET_REGRESSION
        + TARGET_CLASSIFICATION_PDI
        + TARGET_CLASSIFICATION_EE
        + [TARGET_TOXIC]
        + TARGET_BIODIST
    )
    
    # 只保留存在的列
    keep_cols = [c for c in keep_cols if c in df.columns]
    
    # 进行 CV 划分
    if scaffold_split:
        logger.info(f"\nPerforming {n_folds}-fold amine-based scaffold CV split (seed={seed})...")
        fold_splits = amine_based_cv_split(df, n_folds=n_folds, seed=seed, amine_col=amine_col)
        split_method = f"Amine-based scaffold (column: {amine_col})"
    else:
        logger.info(f"\nPerforming {n_folds}-fold random CV split (seed={seed})...")
        fold_splits = random_cv_split(df, n_folds=n_folds, seed=seed)
        split_method = "Random shuffle"
    
    # 保存每个 fold
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, split in enumerate(fold_splits):
        fold_dir = output_dir / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # 只保留需要的列
        train_df = split["train"][keep_cols].reset_index(drop=True)
        val_df = split["val"][keep_cols].reset_index(drop=True)
        test_df = split["test"][keep_cols].reset_index(drop=True)
        
        # 保存
        train_df.to_parquet(fold_dir / "train.parquet", index=False)
        val_df.to_parquet(fold_dir / "val.parquet", index=False)
        test_df.to_parquet(fold_dir / "test.parquet", index=False)
        
        logger.success(f"Saved fold {i} to {fold_dir}")
    
    # 保存列名配置
    config_path = output_dir / "feature_columns.txt"
    with open(config_path, "w") as f:
        f.write("# Feature columns configuration\n\n")
        f.write(f"# SMILES\n{SMILES_COL}\n\n")
        f.write(f"# comp token [{len(COMP_COLS)}]\n")
        f.write("\n".join(COMP_COLS) + "\n\n")
        f.write(f"# phys token [{len(phys_cols)}]\n")
        f.write("\n".join(phys_cols) + "\n\n")
        f.write(f"# help token [{len(HELP_COLS)}]\n")
        f.write("\n".join(HELP_COLS) + "\n\n")
        f.write(f"# exp token [{len(exp_cols)}]\n")
        f.write("\n".join(exp_cols) + "\n\n")
        f.write("# Targets\n")
        f.write("## Regression\n")
        f.write("\n".join(TARGET_REGRESSION) + "\n")
        f.write("## PDI classification\n")
        f.write("\n".join(TARGET_CLASSIFICATION_PDI) + "\n")
        f.write("## EE classification\n")
        f.write("\n".join(TARGET_CLASSIFICATION_EE) + "\n")
        f.write("## Toxic\n")
        f.write(f"{TARGET_TOXIC}\n")
        f.write("## Biodistribution\n")
        f.write("\n".join(TARGET_BIODIST) + "\n")
    
    logger.success(f"Saved feature config to {config_path}")
    
    # 打印汇总
    logger.info("\n" + "=" * 60)
    logger.info("CV DATA PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of folds: {n_folds}")
    logger.info(f"Splitting method: {split_method}")
    logger.info(f"Random seed: {seed}")


if __name__ == "__main__":
    app()

