"""处理 cross-validation 数据脚本：将 CV splits 转换为模型所需的 parquet 格式"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import typer
from loguru import logger

from lnp_ml.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR
from lnp_ml.dataset import (
    LNPDatasetConfig,
    COMP_COLS,
    HELP_COLS,
    get_phys_cols,
    get_exp_cols,
    EXP_ONEHOT_SPECS,
    PHYS_ONEHOT_SPECS,
)


app = typer.Typer()


# CV extra_x 列名到模型列名的映射
CV_COL_MAPPING = {
    # Batch_or_individual_or_barcoded -> Sample_organization_type (for Value_name related)
    "Batch_or_individual_or_barcoded_Barcoded": "Batch_or_individual_or_barcoded_Barcoded",
    "Batch_or_individual_or_barcoded_Individual": "Batch_or_individual_or_barcoded_Individual",
    # Helper_lipid_ID_None 不在模型中使用，忽略
}


def load_cv_split(cv_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载单个 CV split 的数据。
    
    Args:
        cv_dir: CV split 目录，如 cv_0/
        
    Returns:
        (train_df, valid_df, test_df) 合并后的 DataFrame
    """
    splits = {}
    for split_name in ["train", "valid", "test"]:
        # 加载主数据（smiles, quantified_delivery）
        main_path = cv_dir / f"{split_name}.csv"
        extra_x_path = cv_dir / f"{split_name}_extra_x.csv"
        metadata_path = cv_dir / f"{split_name}_metadata.csv"
        
        if not main_path.exists():
            raise FileNotFoundError(f"Missing {main_path}")
        
        main_df = pd.read_csv(main_path)
        
        # 加载 extra_x（已 one-hot 编码的特征）
        if extra_x_path.exists():
            extra_x_df = pd.read_csv(extra_x_path)
            # 确保行数一致
            assert len(main_df) == len(extra_x_df), f"Row count mismatch: {len(main_df)} vs {len(extra_x_df)}"
            # 合并（按行索引）
            df = pd.concat([main_df, extra_x_df], axis=1)
        else:
            df = main_df
            logger.warning(f"  {split_name}_extra_x.csv not found, using main data only")
        
        # 可选：从 metadata 获取额外信息
        if metadata_path.exists():
            metadata_df = pd.read_csv(metadata_path)
            # 提取需要的列（如 Purity, Mix_type, Value_name 等）
            for col in ["Purity", "Mix_type", "Value_name", "Target_or_delivered_gene"]:
                if col in metadata_df.columns and col not in df.columns:
                    df[col] = metadata_df[col]
        
        splits[split_name] = df
    
    return splits["train"], splits["valid"], splits["test"]


def process_cv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理 CV 数据的 DataFrame，对齐到模型所需的列格式。
    
    CV 数据的 extra_x 已经包含大部分 one-hot 编码，但需要：
    1. 添加缺失的 one-hot 列（设为 0）
    2. 从 metadata 中生成 phys token 的 one-hot 列（Purity, Mix_type, Cargo_type, Target_or_delivered_gene）
    3. 生成 Value_name 的 one-hot 列
    """
    df = df.copy()
    
    # 1. 处理 comp 列
    for col in COMP_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    
    # 2. 处理 help 列
    for col in HELP_COLS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0).astype(float)
    
    # 3. 处理 phys token 的 one-hot 列
    for col, values in PHYS_ONEHOT_SPECS.items():
        for v in values:
            onehot_col = f"{col}_{v}"
            if onehot_col not in df.columns:
                # 尝试从原始列生成
                if col in df.columns:
                    df[onehot_col] = (df[col] == v).astype(float)
                else:
                    df[onehot_col] = 0.0
            else:
                df[onehot_col] = df[onehot_col].fillna(0.0).astype(float)
    
    # 4. 处理 exp token 的 one-hot 列
    for col, values in EXP_ONEHOT_SPECS.items():
        for v in values:
            onehot_col = f"{col}_{v}"
            if onehot_col not in df.columns:
                # 尝试从原始列生成
                if col in df.columns:
                    df[onehot_col] = (df[col] == v).astype(float)
                else:
                    df[onehot_col] = 0.0
            else:
                df[onehot_col] = df[onehot_col].fillna(0.0).astype(float)
    
    # 5. 处理 quantified_delivery
    if "quantified_delivery" in df.columns:
        df["quantified_delivery"] = pd.to_numeric(df["quantified_delivery"], errors="coerce")
    
    return df


def get_feature_columns() -> List[str]:
    """获取所有特征列名"""
    config = LNPDatasetConfig()
    return (
        ["smiles"]
        + config.comp_cols
        + config.phys_cols
        + config.help_cols
        + config.exp_cols
        + ["quantified_delivery"]
    )


@app.command()
def main(
    data_dir: Path = EXTERNAL_DATA_DIR / "all_amine_split_for_LiON",
    output_dir: Path = PROCESSED_DATA_DIR / "pretrain_cv",
    n_folds: int = 5,
):
    """
    处理 cross-validation 数据，生成模型所需的 parquet 文件。
    
    输出结构:
        - processed/pretrain_cv/fold_0/train.parquet
        - processed/pretrain_cv/fold_0/valid.parquet
        - processed/pretrain_cv/fold_0/test.parquet
        - processed/pretrain_cv/fold_1/...
        - processed/pretrain_cv/feature_columns.txt
    """
    logger.info(f"Processing CV data from {data_dir}")
    
    # 获取所有 cv_* 目录
    cv_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("cv_")])
    
    if len(cv_dirs) == 0:
        logger.error(f"No cv_* directories found in {data_dir}")
        raise typer.Exit(1)
    
    if len(cv_dirs) != n_folds:
        logger.warning(f"Expected {n_folds} folds, found {len(cv_dirs)}")
    
    logger.info(f"Found {len(cv_dirs)} folds: {[d.name for d in cv_dirs]}")
    
    feature_cols = get_feature_columns()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, cv_dir in enumerate(cv_dirs):
        fold_name = f"fold_{i}"
        fold_output_dir = output_dir / fold_name
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing {cv_dir.name} -> {fold_name}")
        
        # 加载数据
        train_df, valid_df, test_df = load_cv_split(cv_dir)
        
        logger.info(f"  Loaded: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
        
        # 处理数据
        train_df = process_cv_dataframe(train_df)
        valid_df = process_cv_dataframe(valid_df)
        test_df = process_cv_dataframe(test_df)
        
        # 确保所有列存在
        for col in feature_cols:
            for df in [train_df, valid_df, test_df]:
                if col not in df.columns:
                    df[col] = 0.0 if col != "smiles" else ""
        
        # 只保留需要的列
        train_df = train_df[feature_cols]
        valid_df = valid_df[feature_cols]
        test_df = test_df[feature_cols]
        
        # 保存
        train_df.to_parquet(fold_output_dir / "train.parquet", index=False)
        valid_df.to_parquet(fold_output_dir / "valid.parquet", index=False)
        test_df.to_parquet(fold_output_dir / "test.parquet", index=False)
        
        logger.success(f"  Saved to {fold_output_dir}")
    
    # 保存特征列配置
    cols_path = output_dir / "feature_columns.txt"
    with open(cols_path, "w") as f:
        f.write("\n".join(feature_cols))
    logger.success(f"Saved feature columns to {cols_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("CV DATA PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of folds: {len(cv_dirs)}")


if __name__ == "__main__":
    app()

