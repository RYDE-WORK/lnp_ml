"""数据清洗脚本：修正原始数据中的问题"""

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

from lnp_ml.config import RAW_DATA_DIR, INTERIM_DATA_DIR


app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "internal_deleted_uncorrected.xlsx",
    output_path: Path = INTERIM_DATA_DIR / "internal_corrected.csv",
):
    """
    清洗原始数据，修正已知问题。
    
    修正内容：
    1. 修正肌肉注射组 Biodistribution_muscle=0.7745 的数据
    2. 修复阳性对照组 (Amine="Crtl") 的数据
    3. 按给药途径分组进行 z-score 标准化
    4. 对 size 列取 log
    """
    logger.info(f"Loading data from {input_path}")
    df = pd.read_excel(input_path, header=2)
    logger.info(f"Loaded {len(df)} samples")

    # 修正肌肉注射组 0.7745 的数据
    logger.info("Correcting Biodistribution_muscle=0.7745 rows...")
    rows_to_correct = df[df["Biodistribution_muscle"] == 0.7745]
    for index, row in rows_to_correct.iterrows():
        total_biodistribution = pd.to_numeric(row[[
            "Biodistribution_lymph_nodes",
            "Biodistribution_heart",
            "Biodistribution_liver",
            "Biodistribution_spleen",
            "Biodistribution_lung",
            "Biodistribution_kidney",
            "Biodistribution_muscle"
        ]]).sum()
        df.at[index, "Biodistribution_lymph_nodes"] = pd.to_numeric(row["Biodistribution_lymph_nodes"]) / total_biodistribution
        df.at[index, "Biodistribution_heart"] = pd.to_numeric(row["Biodistribution_heart"]) / total_biodistribution
        df.at[index, "Biodistribution_liver"] = pd.to_numeric(row["Biodistribution_liver"]) / total_biodistribution
        df.at[index, "Biodistribution_spleen"] = pd.to_numeric(row["Biodistribution_spleen"]) / total_biodistribution
        df.at[index, "Biodistribution_lung"] = pd.to_numeric(row["Biodistribution_lung"]) / total_biodistribution
        df.at[index, "Biodistribution_kidney"] = pd.to_numeric(row["Biodistribution_kidney"]) / total_biodistribution
        df.at[index, "Biodistribution_muscle"] = pd.to_numeric(row["Biodistribution_muscle"]) / total_biodistribution
        df.at[index, "quantified_total_luminescence"] = pd.to_numeric(row["quantified_total_luminescence"]) / (1 - 0.7745)
        df.at[index, "unnormalized_delivery"] = df.at[index, "quantified_total_luminescence"]
    logger.info(f"  Corrected {len(rows_to_correct)} rows")

    # 修复阳性对照组的数据
    logger.info("Fixing control group (Amine='Crtl')...")
    rows_to_override = df["Amine"] == "Crtl"
    df.loc[rows_to_override, "quantified_total_luminescence"] = 1
    df.loc[rows_to_override, "unnormalized_delivery"] = 1
    logger.info(f"  Fixed {rows_to_override.sum()} rows")

    # 分别对肌肉注射组和静脉注射组重新进行 z-score 标准化
    logger.info("Z-score normalizing delivery by Route_of_administration...")
    df["unnormalized_delivery"] = pd.to_numeric(df["unnormalized_delivery"], errors="coerce")
    df["quantified_delivery"] = (
        df.groupby("Route_of_administration")["unnormalized_delivery"]
          .transform(lambda x: (x - x.mean()) / x.std())
    )

    # 对 size 列取 log
    logger.info("Log-transforming size column...")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df["size"] = np.log(df["size"].replace(0, np.nan))  # 避免 log(0)

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Saved cleaned data to {output_path}")


if __name__ == "__main__":
    app()
