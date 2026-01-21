"""预测脚本：使用训练好的模型进行推理"""

from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader
from loguru import logger
import typer

from lnp_ml.config import MODELS_DIR, PROCESSED_DATA_DIR
from lnp_ml.dataset import LNPDataset, collate_fn
from lnp_ml.modeling.models import LNPModel, LNPModelWithoutMPNN


app = typer.Typer()

# MPNN ensemble 默认路径
DEFAULT_MPNN_ENSEMBLE_DIR = MODELS_DIR / "mpnn" / "all_amine_split_for_LiON"


def find_mpnn_ensemble_paths(base_dir: Path = DEFAULT_MPNN_ENSEMBLE_DIR) -> List[str]:
    """自动查找 MPNN ensemble 的 model.pt 文件。"""
    model_paths = sorted(base_dir.glob("cv_*/fold_*/model_*/model.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No model.pt files found in {base_dir}")
    return [str(p) for p in model_paths]


def load_model(
    model_path: Path,
    device: torch.device,
    mpnn_device: str = "cpu",
) -> Union[LNPModel, LNPModelWithoutMPNN]:
    """
    加载训练好的模型。
    
    自动根据 checkpoint 的 config.use_mpnn 选择模型类型。
    """
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]
    use_mpnn = config.get("use_mpnn", False)
    
    if use_mpnn:
        # 总是自动查找 MPNN ensemble，避免使用 checkpoint 中的旧绝对路径（可能来自其他机器）
        logger.info("Model was trained with MPNN, auto-detecting ensemble...")
        ensemble_paths = find_mpnn_ensemble_paths()
        logger.info(f"Found {len(ensemble_paths)} MPNN models")
        
        model = LNPModel(
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            n_attn_layers=config["n_attn_layers"],
            fusion_strategy=config["fusion_strategy"],
            head_hidden_dim=config["head_hidden_dim"],
            dropout=config["dropout"],
            mpnn_ensemble_paths=ensemble_paths,
            mpnn_device=mpnn_device,
        )
    else:
        model = LNPModelWithoutMPNN(
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            n_attn_layers=config["n_attn_layers"],
            fusion_strategy=config["fusion_strategy"],
            head_hidden_dim=config["head_hidden_dim"],
            dropout=config["dropout"],
        )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Model config: {config}")
    logger.info(f"Best val_loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    return model


@torch.no_grad()
def predict_batch(
    model: Union[LNPModel, LNPModelWithoutMPNN],
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, List]:
    """对整个数据集进行预测"""
    model.eval()
    
    all_preds = {
        "size": [],
        "pdi": [],
        "ee": [],
        "delivery": [],
        "biodist": [],
        "toxic": [],
    }
    all_smiles = []
    
    for batch in loader:
        smiles = batch["smiles"]
        tabular = {k: v.to(device) for k, v in batch["tabular"].items()}
        
        outputs = model(smiles, tabular)
        
        all_smiles.extend(smiles)
        
        # 回归任务
        all_preds["size"].extend(outputs["size"].squeeze(-1).cpu().tolist())
        all_preds["delivery"].extend(outputs["delivery"].squeeze(-1).cpu().tolist())
        
        # 分类任务：取 argmax
        all_preds["pdi"].extend(outputs["pdi"].argmax(dim=-1).cpu().tolist())
        all_preds["ee"].extend(outputs["ee"].argmax(dim=-1).cpu().tolist())
        all_preds["toxic"].extend(outputs["toxic"].argmax(dim=-1).cpu().tolist())
        
        # 分布任务
        all_preds["biodist"].extend(outputs["biodist"].cpu().tolist())
    
    return {"smiles": all_smiles, **all_preds}


def predictions_to_dataframe(predictions: Dict) -> pd.DataFrame:
    """将预测结果转换为 DataFrame"""
    # 基本列
    df = pd.DataFrame({
        "smiles": predictions["smiles"],
        "pred_size": predictions["size"],
        "pred_delivery": predictions["delivery"],
        "pred_pdi_class": predictions["pdi"],
        "pred_ee_class": predictions["ee"],
        "pred_toxic": predictions["toxic"],
    })
    
    # PDI 类别映射
    pdi_labels = ["0_0to0_2", "0_2to0_3", "0_3to0_4", "0_4to0_5"]
    df["pred_pdi_label"] = df["pred_pdi_class"].map(lambda x: pdi_labels[x])
    
    # EE 类别映射
    ee_labels = ["EE<50", "50<=EE<80", "80<EE<=100"]
    df["pred_ee_label"] = df["pred_ee_class"].map(lambda x: ee_labels[x])
    
    # Biodistribution 展开为多列
    biodist_cols = [
        "pred_biodist_lymph_nodes",
        "pred_biodist_heart",
        "pred_biodist_liver",
        "pred_biodist_spleen",
        "pred_biodist_lung",
        "pred_biodist_kidney",
        "pred_biodist_muscle",
    ]
    biodist_df = pd.DataFrame(predictions["biodist"], columns=biodist_cols)
    df = pd.concat([df, biodist_df], axis=1)
    
    return df


@app.command()
def main(
    test_path: Path = PROCESSED_DATA_DIR / "test.parquet",
    model_path: Path = MODELS_DIR / "model.pt",
    output_path: Path = PROCESSED_DATA_DIR / "predictions.parquet",
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    使用训练好的模型进行预测。
    """
    logger.info(f"Using device: {device}")
    device = torch.device(device)
    
    # 加载模型
    model = load_model(model_path, device)
    
    # 加载数据
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_parquet(test_path)
    test_dataset = LNPDataset(test_df)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # 预测
    logger.info("Running predictions...")
    predictions = predict_batch(model, test_loader, device)
    
    # 转换为 DataFrame
    pred_df = predictions_to_dataframe(predictions)
    
    # 保存
    pred_df.to_parquet(output_path, index=False)
    logger.success(f"Saved predictions to {output_path}")
    
    # 打印统计
    logger.info("\n=== Prediction Statistics ===")
    logger.info(f"Total samples: {len(pred_df)}")
    logger.info(f"\nSize (pred): mean={pred_df['pred_size'].mean():.4f}, std={pred_df['pred_size'].std():.4f}")
    logger.info(f"Delivery (pred): mean={pred_df['pred_delivery'].mean():.4f}, std={pred_df['pred_delivery'].std():.4f}")
    logger.info(f"\nPDI class distribution:\n{pred_df['pred_pdi_label'].value_counts()}")
    logger.info(f"\nEE class distribution:\n{pred_df['pred_ee_label'].value_counts()}")
    logger.info(f"\nToxic distribution:\n{pred_df['pred_toxic'].value_counts()}")


@app.command()
def test(
    test_path: Path = PROCESSED_DATA_DIR / "test.parquet",
    model_path: Path = MODELS_DIR / "model.pt",
    output_path: Path = PROCESSED_DATA_DIR / "test_results.json",
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    在测试集上完整评估模型性能，输出详细指标。
    """
    import json
    import numpy as np
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        accuracy_score,
        classification_report,
    )
    from lnp_ml.modeling.trainer import validate
    
    logger.info(f"Using device: {device}")
    device_obj = torch.device(device)
    
    # 加载模型
    model = load_model(model_path, device_obj)
    
    # 加载数据
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_parquet(test_path)
    test_dataset = LNPDataset(test_df)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # 基础损失指标
    logger.info("Computing loss metrics...")
    loss_metrics = validate(model, test_loader, device_obj)
    
    # 获取预测和真实值
    logger.info("Computing detailed metrics...")
    predictions = predict_batch(model, test_loader, device_obj)
    
    results = {"loss_metrics": loss_metrics, "detailed_metrics": {}}
    
    # 回归指标：size
    if "size" in test_df.columns:
        mask = ~test_df["size"].isna()
        if mask.any():
            y_true = test_df.loc[mask, "size"].values
            y_pred = np.array(predictions["size"])[mask.values]
            results["detailed_metrics"]["size"] = {
                "mse": float(mean_squared_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
            }
    
    # 回归指标：delivery
    if "quantified_delivery" in test_df.columns:
        mask = ~test_df["quantified_delivery"].isna()
        if mask.any():
            y_true = test_df.loc[mask, "quantified_delivery"].values
            y_pred = np.array(predictions["delivery"])[mask.values]
            results["detailed_metrics"]["delivery"] = {
                "mse": float(mean_squared_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
            }
    
    # 分类指标：PDI
    pdi_cols = ["PDI_0_0to0_2", "PDI_0_2to0_3", "PDI_0_3to0_4", "PDI_0_4to0_5"]
    if all(c in test_df.columns for c in pdi_cols):
        pdi_true = test_df[pdi_cols].values.argmax(axis=1)
        mask = test_df[pdi_cols].sum(axis=1) > 0
        if mask.any():
            y_true = pdi_true[mask]
            y_pred = np.array(predictions["pdi"])[mask]
            results["detailed_metrics"]["pdi"] = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
            }
    
    # 分类指标：EE
    ee_cols = ["Encapsulation_Efficiency_EE<50", "Encapsulation_Efficiency_50<=EE<80", "Encapsulation_Efficiency_80<EE<=100"]
    if all(c in test_df.columns for c in ee_cols):
        ee_true = test_df[ee_cols].values.argmax(axis=1)
        mask = test_df[ee_cols].sum(axis=1) > 0
        if mask.any():
            y_true = ee_true[mask]
            y_pred = np.array(predictions["ee"])[mask]
            results["detailed_metrics"]["ee"] = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
            }
    
    # 分类指标：toxic
    if "toxic" in test_df.columns:
        mask = test_df["toxic"].notna() & (test_df["toxic"] >= 0)
        if mask.any():
            y_true = test_df.loc[mask, "toxic"].astype(int).values
            y_pred = np.array(predictions["toxic"])[mask.values]
            results["detailed_metrics"]["toxic"] = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
            }
    
    # 打印结果
    logger.info("\n" + "=" * 50)
    logger.info("TEST RESULTS")
    logger.info("=" * 50)
    
    logger.info("\n[Loss Metrics]")
    for k, v in loss_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    logger.info("\n[Detailed Metrics]")
    for task, metrics in results["detailed_metrics"].items():
        logger.info(f"\n  {task}:")
        for k, v in metrics.items():
            logger.info(f"    {k}: {v:.4f}")
    
    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"\nSaved test results to {output_path}")


# 保留旧的 evaluate 作为 test 的别名
@app.command()
def evaluate(
    test_path: Path = PROCESSED_DATA_DIR / "test.parquet",
    model_path: Path = MODELS_DIR / "model.pt",
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    [已废弃] 请使用 'test' 命令。
    """
    test(test_path, model_path, PROCESSED_DATA_DIR / "test_results.json", batch_size, device)


if __name__ == "__main__":
    app()
