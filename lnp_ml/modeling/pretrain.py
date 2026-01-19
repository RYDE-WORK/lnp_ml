"""预训练脚本：在外部 LiON 数据上预训练 backbone + delivery head"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm
import typer

from lnp_ml.config import MODELS_DIR, PROCESSED_DATA_DIR
from lnp_ml.dataset import ExternalDeliveryDataset, collate_fn

# MPNN ensemble 默认路径
DEFAULT_MPNN_ENSEMBLE_DIR = MODELS_DIR / "mpnn" / "all_amine_split_for_LiON"


def find_mpnn_ensemble_paths(base_dir: Path = DEFAULT_MPNN_ENSEMBLE_DIR) -> List[str]:
    """
    自动查找 MPNN ensemble 的 model.pt 文件。
    
    在 base_dir 下查找所有 cv_*/fold_*/model_*/model.pt 文件。
    """
    model_paths = sorted(base_dir.glob("cv_*/fold_*/model_*/model.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No model.pt files found in {base_dir}")
    return [str(p) for p in model_paths]
from lnp_ml.modeling.models import LNPModel, LNPModelWithoutMPNN


app = typer.Typer()


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def warmup_cache(model: nn.Module, smiles_list: List[str], batch_size: int = 256) -> None:
    """
    预热 RDKit 特征缓存，避免训练时计算阻塞。
    """
    unique_smiles = list(set(smiles_list))
    logger.info(f"Warming up RDKit cache for {len(unique_smiles)} unique SMILES...")
    
    for i in tqdm(range(0, len(unique_smiles), batch_size), desc="Cache warmup"):
        batch = unique_smiles[i:i + batch_size]
        model.rdkit_encoder(batch)
    
    logger.success(f"Cache warmup complete. Cached {len(model.rdkit_encoder._cache)} SMILES.")


def train_epoch_delivery(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
) -> Dict[str, float]:
    """
    单个 epoch 的预训练（仅 delivery 任务）。
    """
    model.train()
    total_loss = 0.0
    n_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for batch in pbar:
        smiles = batch["smiles"]
        tabular = {k: v.to(device) for k, v in batch["tabular"].items()}
        targets = batch["targets"]["delivery"].to(device)
        mask = batch["mask"]["delivery"].to(device)

        optimizer.zero_grad()

        # Forward: 只预测 delivery
        pred = model.forward_delivery(smiles, tabular)  # [B, 1]
        pred = pred.squeeze(-1)  # [B]

        # 计算损失（仅对有效样本）
        if mask.any():
            loss = nn.functional.mse_loss(pred[mask], targets[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mask.sum().item()
            n_samples += mask.sum().item()
            
            pbar.set_postfix({"loss": total_loss / max(n_samples, 1)})

    avg_loss = total_loss / max(n_samples, 1)
    return {"loss": avg_loss, "n_samples": n_samples}


@torch.no_grad()
def validate_delivery(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    验证（仅 delivery 任务）。
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        smiles = batch["smiles"]
        tabular = {k: v.to(device) for k, v in batch["tabular"].items()}
        targets = batch["targets"]["delivery"].to(device)
        mask = batch["mask"]["delivery"].to(device)

        pred = model.forward_delivery(smiles, tabular).squeeze(-1)

        if mask.any():
            loss = nn.functional.mse_loss(pred[mask], targets[mask])
            total_loss += loss.item() * mask.sum().item()
            n_samples += mask.sum().item()

    avg_loss = total_loss / max(n_samples, 1)
    return {"loss": avg_loss, "n_samples": n_samples}


def pretrain(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    epochs: int = 50,
    patience: int = 10,
) -> dict:
    """
    预训练循环。

    Returns:
        训练历史和最佳验证损失
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    early_stopping = EarlyStopping(patience=patience)

    history = {"train": [], "val": []}
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        # Train
        train_metrics = train_epoch_delivery(model, train_loader, optimizer, device, epoch)

        # Validate
        val_metrics = validate_delivery(model, val_loader, device)

        # Log
        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}"
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Learning rate scheduling
        scheduler.step(val_metrics["loss"])

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  -> New best model (val_loss={best_val_loss:.4f})")

        # Early stopping
        if early_stopping(val_metrics["loss"]):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_val_loss": best_val_loss,
    }


@app.command()
def main(
    # 数据路径（已处理的 parquet 文件）
    train_path: Path = PROCESSED_DATA_DIR / "train_pretrain.parquet",
    val_path: Path = PROCESSED_DATA_DIR / "val_pretrain.parquet",
    output_dir: Path = MODELS_DIR,
    # 模型参数
    d_model: int = 256,
    num_heads: int = 8,
    n_attn_layers: int = 4,
    fusion_strategy: str = "attention",
    head_hidden_dim: int = 128,
    dropout: float = 0.1,
    # MPNN 参数（可选）
    use_mpnn: bool = False,  # 启用 MPNN，自动从默认路径加载 ensemble
    mpnn_checkpoint: Optional[str] = None,
    mpnn_ensemble_paths: Optional[str] = None,  # 逗号分隔的路径列表
    mpnn_device: str = "cpu",
    # 训练参数
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    epochs: int = 50,
    patience: int = 10,
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    在外部 LiON 数据上预训练 backbone + delivery head。

    需要先运行 `make data_pretrain` 生成 parquet 文件。
    
    使用 --use-mpnn 启用 MPNN encoder（自动从 models/mpnn/all_amine_split_for_LiON 加载）。

    产出:
        - models/pretrain_delivery.pt: 包含 backbone + delivery head 权重
        - models/pretrain_history.json: 训练历史
    """
    logger.info(f"Using device: {device}")
    device_obj = torch.device(device)

    # 加载已处理的 parquet 文件
    logger.info(f"Loading train data from {train_path}")
    train_df = pd.read_parquet(train_path)
    train_dataset = ExternalDeliveryDataset(train_df)
    
    logger.info(f"Loading val data from {val_path}")
    val_df = pd.read_parquet(val_path)
    val_dataset = ExternalDeliveryDataset(val_df)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # 解析 MPNN 配置
    # 优先级：mpnn_checkpoint > mpnn_ensemble_paths > use_mpnn（自动查找）
    ensemble_paths_list = None
    if mpnn_ensemble_paths:
        ensemble_paths_list = mpnn_ensemble_paths.split(",")
    elif use_mpnn and mpnn_checkpoint is None:
        # --use-mpnn 但没有指定具体路径，自动查找
        logger.info(f"Auto-detecting MPNN ensemble from {DEFAULT_MPNN_ENSEMBLE_DIR}")
        ensemble_paths_list = find_mpnn_ensemble_paths()
        logger.info(f"Found {len(ensemble_paths_list)} MPNN models")
    
    enable_mpnn = mpnn_checkpoint is not None or ensemble_paths_list is not None

    # 创建模型
    logger.info(f"Creating model (use_mpnn={enable_mpnn})...")
    if enable_mpnn:
        model = LNPModel(
            d_model=d_model,
            num_heads=num_heads,
            n_attn_layers=n_attn_layers,
            fusion_strategy=fusion_strategy,
            head_hidden_dim=head_hidden_dim,
            dropout=dropout,
            mpnn_checkpoint=mpnn_checkpoint,
            mpnn_ensemble_paths=ensemble_paths_list,
            mpnn_device=mpnn_device,
        )
    else:
        model = LNPModelWithoutMPNN(
            d_model=d_model,
            num_heads=num_heads,
            n_attn_layers=n_attn_layers,
            fusion_strategy=fusion_strategy,
            head_hidden_dim=head_hidden_dim,
            dropout=dropout,
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # 预热 RDKit 缓存（避免训练时阻塞）
    all_smiles = train_df["smiles"].tolist() + val_df["smiles"].tolist()
    warmup_cache(model, all_smiles, batch_size=256)

    # 预训练
    logger.info("Starting pretraining on external data (delivery only)...")
    result = pretrain(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=device_obj,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=patience,
    )

    # 保存预训练 checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "pretrain_delivery.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "backbone_state_dict": model.get_backbone_state_dict(),
            "delivery_head_state_dict": model.get_delivery_head_state_dict(),
            "config": {
                "d_model": d_model,
                "num_heads": num_heads,
                "n_attn_layers": n_attn_layers,
                "fusion_strategy": fusion_strategy,
                "head_hidden_dim": head_hidden_dim,
                "dropout": dropout,
                "use_mpnn": enable_mpnn,
            },
            "best_val_loss": result["best_val_loss"],
        },
        checkpoint_path,
    )
    logger.success(f"Saved pretrain checkpoint to {checkpoint_path}")

    # 保存训练历史
    history_path = output_dir / "pretrain_history.json"
    with open(history_path, "w") as f:
        json.dump(result["history"], f, indent=2)
    logger.success(f"Saved pretrain history to {history_path}")

    logger.success(
        f"Pretraining complete! Best val_loss: {result['best_val_loss']:.4f}"
    )


@app.command()
def test(
    # 数据路径
    val_path: Path = PROCESSED_DATA_DIR / "val_pretrain.parquet",
    model_path: Path = MODELS_DIR / "pretrain_delivery.pt",
    output_path: Path = MODELS_DIR / "pretrain_test_results.json",
    # MPNN 参数
    use_mpnn: bool = False,
    mpnn_device: str = "cpu",
    # 其他参数
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    评估 pretrain 模型在外部数据上的 delivery 预测性能。
    
    输出详细指标：MSE, RMSE, MAE, R²
    """
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    logger.info(f"Using device: {device}")
    device_obj = torch.device(device)
    
    # 加载模型
    logger.info(f"Loading pretrain model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device_obj)
    config = checkpoint["config"]
    
    # 解析 MPNN 配置
    enable_mpnn = config.get("use_mpnn", False)
    if enable_mpnn or use_mpnn:
        logger.info(f"Auto-detecting MPNN ensemble from {DEFAULT_MPNN_ENSEMBLE_DIR}")
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
    model.to(device_obj)
    model.eval()
    
    logger.info(f"Model config: {config}")
    logger.info(f"Best val_loss from training: {checkpoint.get('best_val_loss', 'N/A')}")
    
    # 加载数据
    logger.info(f"Loading validation data from {val_path}")
    val_df = pd.read_parquet(val_path)
    val_dataset = ExternalDeliveryDataset(val_df)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # 预测
    logger.info("Running predictions...")
    all_preds = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Predicting"):
            smiles = batch["smiles"]
            tabular = {k: v.to(device_obj) for k, v in batch["tabular"].items()}
            targets = batch["targets"]["delivery"].numpy()
            mask = batch["mask"]["delivery"].numpy()
            
            pred = model.forward_delivery(smiles, tabular).squeeze(-1).cpu().numpy()
            
            all_preds.extend(pred)
            all_targets.extend(targets)
            all_masks.extend(mask)
    
    # 转为数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_masks = np.array(all_masks, dtype=bool)
    
    # 只计算有效样本
    y_pred = all_preds[all_masks]
    y_true = all_targets[all_masks]
    
    # 计算指标
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    # 额外统计
    correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
    
    results = {
        "model_path": str(model_path),
        "val_path": str(val_path),
        "n_samples": int(all_masks.sum()),
        "metrics": {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "correlation": correlation,
        },
        "statistics": {
            "y_true_mean": float(y_true.mean()),
            "y_true_std": float(y_true.std()),
            "y_pred_mean": float(y_pred.mean()),
            "y_pred_std": float(y_pred.std()),
        }
    }
    
    # 打印结果
    logger.info("\n" + "=" * 50)
    logger.info("PRETRAIN MODEL EVALUATION (Delivery)")
    logger.info("=" * 50)
    logger.info(f"Samples: {results['n_samples']}")
    logger.info("\n[Metrics]")
    logger.info(f"  MSE:         {mse:.4f}")
    logger.info(f"  RMSE:        {rmse:.4f}")
    logger.info(f"  MAE:         {mae:.4f}")
    logger.info(f"  R²:          {r2:.4f}")
    logger.info(f"  Correlation: {correlation:.4f}")
    logger.info("\n[Statistics]")
    logger.info(f"  True:  mean={y_true.mean():.4f}, std={y_true.std():.4f}")
    logger.info(f"  Pred:  mean={y_pred.mean():.4f}, std={y_pred.std():.4f}")
    
    # 保存结果
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    app()

