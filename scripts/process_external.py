"""外部数据预处理脚本：external -> processed"""

from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from sklearn.model_selection import train_test_split

from lnp_ml.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR
from lnp_ml.dataset import process_external_dataframe, LNPDatasetConfig, get_phys_cols, get_exp_cols, COMP_COLS, HELP_COLS


app = typer.Typer()


@app.command()
def main(
    input_path: Path = EXTERNAL_DATA_DIR / "all_data_LiON.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
    train_ratio: float = 0.9,
    seed: int = 42,
):
    """
    处理外部 LiON 数据，生成预训练用的 parquet 文件。
    
    输出:
        - processed/train_pretrain.parquet
        - processed/val_pretrain.parquet
        - processed/feature_columns_pretrain.txt
    """
    logger.info(f"Loading external data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # 过滤掉 quantified_delivery 为空的行
    if "quantified_delivery" in df.columns:
        before_len = len(df)
        df = df[df["quantified_delivery"].notna()].reset_index(drop=True)
        logger.info(f"Filtered NaN delivery: {before_len} -> {len(df)} samples")
    
    # 处理数据（列对齐、one-hot 生成）
    logger.info("Processing dataframe (column alignment, one-hot encoding)...")
    df = process_external_dataframe(df)
    
    # 获取所需列
    config = LNPDatasetConfig()
    feature_cols = (
        ["smiles"]
        + config.comp_cols
        + config.phys_cols
        + config.help_cols
        + config.exp_cols
        + ["quantified_delivery"]
    )
    
    # 只保留需要的列
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns (will be filled with 0): {missing_cols}")
        for col in missing_cols:
            df[col] = 0.0
    
    df = df[feature_cols]
    
    # 划分 train/val
    logger.info(f"Splitting data: train_ratio={train_ratio}, seed={seed}")
    train_df, val_df = train_test_split(
        df, train_size=train_ratio, random_state=seed, shuffle=True
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train_pretrain.parquet"
    val_path = output_dir / "val_pretrain.parquet"
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    
    logger.success(f"Saved train data to {train_path}")
    logger.success(f"Saved val data to {val_path}")
    
    # 保存特征列配置
    cols_path = output_dir / "feature_columns_pretrain.txt"
    with open(cols_path, "w") as f:
        f.write("\n".join(feature_cols))
    logger.success(f"Saved feature columns to {cols_path}")


if __name__ == "__main__":
    app()

