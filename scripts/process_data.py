"""数据处理脚本：将原始数据转换为模型可用的格式"""

from pathlib import Path

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


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "internal_corrected.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
    train_ratio: float = 0.56,
    val_ratio: float = 0.14,
    seed: int = 42,
):
    """
    处理原始数据并划分训练/验证/测试集。
    
    输出文件:
        - train.parquet: 训练集
        - val.parquet: 验证集
        - test.parquet: 测试集
        - feature_columns.txt: 特征列名配置
    """
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # 处理数据
    logger.info("Processing dataframe...")
    df = process_dataframe(df)
    
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
    df = df[keep_cols]
    
    # 随机打乱并划分
    logger.info("Splitting dataset...")
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    logger.success(f"Saved train to {train_path}")
    logger.success(f"Saved val to {val_path}")
    logger.success(f"Saved test to {test_path}")
    
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
    
    # 打印统计信息
    logger.info("\n=== Dataset Statistics ===")
    logger.info(f"Total samples: {n}")
    logger.info(f"SMILES unique: {df[SMILES_COL].nunique()}")
    
    # 缺失值统计
    logger.info("\nMissing values in targets:")
    for col in TARGET_REGRESSION + [TARGET_TOXIC]:
        if col in df.columns:
            missing = df[col].isna().sum()
            logger.info(f"  {col}: {missing} ({100*missing/n:.1f}%)")
    
    # PDI 分布
    if all(c in df.columns for c in TARGET_CLASSIFICATION_PDI):
        pdi_sum = df[TARGET_CLASSIFICATION_PDI].sum()
        logger.info(f"\nPDI distribution:")
        for col, count in pdi_sum.items():
            logger.info(f"  {col}: {int(count)}")
    
    # EE 分布
    if all(c in df.columns for c in TARGET_CLASSIFICATION_EE):
        ee_sum = df[TARGET_CLASSIFICATION_EE].sum()
        logger.info(f"\nEE distribution:")
        for col, count in ee_sum.items():
            logger.info(f"  {col}: {int(count)}")


if __name__ == "__main__":
    app()

