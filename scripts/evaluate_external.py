"""评估外部数据 cross-validation 结果的脚本"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import mean_squared_error, r2_score

from lnp_ml.config import EXTERNAL_DATA_DIR


app = typer.Typer()


def evaluate_split(
    test_path: Path,
    preds_path: Path,
    y_col: str = "quantified_delivery",
) -> Dict[str, float]:
    """
    评估单个 split 的预测结果。
    
    Args:
        test_path: groundtruth CSV 路径（包含 smiles 和 y_col）
        preds_path: 预测结果 CSV 路径（包含 smiles 和 y_col）
        y_col: Y 值列名
        
    Returns:
        Dict with rmse, r2, mse, mae, n_samples
    """
    # 读取数据
    test_df = pd.read_csv(test_path)
    preds_df = pd.read_csv(preds_path)
    
    # 检查列是否存在
    if y_col not in test_df.columns:
        raise ValueError(f"Column '{y_col}' not found in {test_path}")
    if y_col not in preds_df.columns:
        raise ValueError(f"Column '{y_col}' not found in {preds_path}")
    
    # 直接按行对齐（test.csv 和 preds.csv 顺序一致）
    assert len(test_df) == len(preds_df), f"行数不匹配: test={len(test_df)}, preds={len(preds_df)}"
    
    # 验证 smiles 列一致（如果存在）
    if "smiles" in test_df.columns and "smiles" in preds_df.columns:
        if not (test_df["smiles"].values == preds_df["smiles"].values).all():
            logger.warning("smiles 列顺序不一致，请检查数据")
    
    y_true = test_df[y_col].values
    y_pred = preds_df[y_col].values
    
    # 计算指标
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
    
    return {
        "n_samples": len(y_true),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "correlation": correlation,
    }


@app.command()
def main(
    data_dir: Path = EXTERNAL_DATA_DIR / "all_amine_split_for_LiON",
    output_path: Path = EXTERNAL_DATA_DIR / "all_amine_split_for_LiON" / "evaluation_results.json",
    y_col: str = "quantified_delivery",
):
    """
    评估 all_amine_split_for_LiON 的 cross-validation 结果。
    
    计算每个 split 和整体的 RMSE、R² 指标。
    """
    logger.info(f"Evaluating cross-validation results from {data_dir}")
    
    # 找到所有 cv_* 目录
    cv_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("cv_")])
    
    if not cv_dirs:
        logger.error(f"No cv_* directories found in {data_dir}")
        raise typer.Exit(1)
    
    logger.info(f"Found {len(cv_dirs)} splits: {[d.name for d in cv_dirs]}")
    
    # 评估每个 split
    split_results = {}
    all_y_true = []
    all_y_pred = []
    
    for cv_dir in cv_dirs:
        split_name = cv_dir.name
        test_path = cv_dir / "test.csv"
        preds_path = cv_dir / "preds.csv"
        
        if not test_path.exists():
            logger.warning(f"  {split_name}: test.csv not found, skipping")
            continue
        if not preds_path.exists():
            logger.warning(f"  {split_name}: preds.csv not found, skipping")
            continue
        
        # 评估
        metrics = evaluate_split(test_path, preds_path, y_col)
        split_results[split_name] = metrics
        
        logger.info(
            f"  {split_name}: n={metrics['n_samples']}, "
            f"RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}"
        )
        
        # 收集所有数据用于计算整体指标
        test_df = pd.read_csv(test_path)
        preds_df = pd.read_csv(preds_path)
        all_y_true.extend(test_df[y_col].tolist())
        all_y_pred.extend(preds_df[y_col].tolist())
    
    # 计算整体指标
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    overall_mse = float(mean_squared_error(all_y_true, all_y_pred))
    overall_rmse = float(np.sqrt(overall_mse))
    overall_r2 = float(r2_score(all_y_true, all_y_pred))
    overall_mae = float(np.mean(np.abs(all_y_true - all_y_pred)))
    overall_correlation = float(np.corrcoef(all_y_true, all_y_pred)[0, 1])
    
    overall_results = {
        "n_samples": len(all_y_true),
        "mse": overall_mse,
        "rmse": overall_rmse,
        "mae": overall_mae,
        "r2": overall_r2,
        "correlation": overall_correlation,
    }
    
    # 计算 split 指标的均值和标准差
    split_metrics = list(split_results.values())
    summary_stats = {}
    for metric in ["rmse", "r2", "mae", "correlation"]:
        values = [s[metric] for s in split_metrics]
        summary_stats[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    
    # 汇总结果
    results = {
        "data_dir": str(data_dir),
        "y_col": y_col,
        "n_splits": len(split_results),
        "split_results": split_results,
        "overall": overall_results,
        "summary_stats": summary_stats,
    }
    
    # 打印结果
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-VALIDATION EVALUATION RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"\n[Per-Split Results]")
    for split_name, metrics in sorted(split_results.items()):
        logger.info(
            f"  {split_name}: RMSE={metrics['rmse']:.4f}, "
            f"R²={metrics['r2']:.4f}, "
            f"MAE={metrics['mae']:.4f}, "
            f"Corr={metrics['correlation']:.4f}"
        )
    
    logger.info(f"\n[Summary Statistics (across {len(split_results)} splits)]")
    for metric, stats in summary_stats.items():
        logger.info(
            f"  {metric.upper():12s}: "
            f"mean={stats['mean']:.4f} ± {stats['std']:.4f} "
            f"(min={stats['min']:.4f}, max={stats['max']:.4f})"
        )
    
    logger.info(f"\n[Overall (all {overall_results['n_samples']} samples pooled)]")
    logger.info(f"  RMSE:        {overall_results['rmse']:.4f}")
    logger.info(f"  R²:          {overall_results['r2']:.4f}")
    logger.info(f"  MAE:         {overall_results['mae']:.4f}")
    logger.info(f"  Correlation: {overall_results['correlation']:.4f}")
    
    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    app()
