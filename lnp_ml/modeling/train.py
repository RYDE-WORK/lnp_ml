"""训练脚本：支持超参数调优"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from loguru import logger
import typer

from lnp_ml.config import MODELS_DIR, PROCESSED_DATA_DIR
from lnp_ml.dataset import LNPDataset, collate_fn
from lnp_ml.modeling.models import LNPModelWithoutMPNN
from lnp_ml.modeling.trainer import (
    train_epoch,
    validate,
    EarlyStopping,
    LossWeights,
)


app = typer.Typer()


def create_model(
    d_model: int = 256,
    num_heads: int = 8,
    n_attn_layers: int = 4,
    fusion_strategy: str = "attention",
    head_hidden_dim: int = 128,
    dropout: float = 0.1,
) -> LNPModelWithoutMPNN:
    """创建模型"""
    return LNPModelWithoutMPNN(
        d_model=d_model,
        num_heads=num_heads,
        n_attn_layers=n_attn_layers,
        fusion_strategy=fusion_strategy,
        head_hidden_dim=head_hidden_dim,
        dropout=dropout,
    )


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    epochs: int = 100,
    patience: int = 15,
    loss_weights: Optional[LossWeights] = None,
) -> dict:
    """
    训练模型。
    
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
        train_metrics = train_epoch(model, train_loader, optimizer, device, loss_weights)
        
        # Validate
        val_metrics = validate(model, val_loader, device, loss_weights)
        
        # Log
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
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
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return {
        "history": history,
        "best_val_loss": best_val_loss,
    }


def run_hyperparameter_tuning(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_trials: int = 20,
    epochs_per_trial: int = 30,
) -> dict:
    """
    使用 Optuna 进行超参数调优。
    
    Returns:
        最佳超参数
    """
    try:
        import optuna
    except ImportError:
        logger.error("Optuna not installed. Run: pip install optuna")
        raise
    
    def objective(trial: optuna.Trial) -> float:
        # 采样超参数
        d_model = trial.suggest_categorical("d_model", [128, 256, 512])
        num_heads = trial.suggest_categorical("num_heads", [4, 8])
        n_attn_layers = trial.suggest_int("n_attn_layers", 2, 6)
        fusion_strategy = trial.suggest_categorical(
            "fusion_strategy", ["attention", "avg", "max"]
        )
        head_hidden_dim = trial.suggest_categorical("head_hidden_dim", [64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.05, 0.3)
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        
        # 创建模型
        model = create_model(
            d_model=d_model,
            num_heads=num_heads,
            n_attn_layers=n_attn_layers,
            fusion_strategy=fusion_strategy,
            head_hidden_dim=head_hidden_dim,
            dropout=dropout,
        )
        
        # 训练
        result = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs_per_trial,
            patience=10,
        )
        
        return result["best_val_loss"]
    
    # 运行优化
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best val_loss: {study.best_trial.value:.4f}")
    logger.info(f"Best params: {study.best_trial.params}")
    
    return study.best_trial.params


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train.parquet",
    val_path: Path = PROCESSED_DATA_DIR / "val.parquet",
    output_dir: Path = MODELS_DIR,
    # 模型参数
    d_model: int = 256,
    num_heads: int = 8,
    n_attn_layers: int = 4,
    fusion_strategy: str = "attention",
    head_hidden_dim: int = 128,
    dropout: float = 0.1,
    # 训练参数
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    epochs: int = 100,
    patience: int = 15,
    # 超参数调优
    tune: bool = False,
    n_trials: int = 20,
    epochs_per_trial: int = 30,
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    训练 LNP 预测模型。
    
    使用 --tune 启用超参数调优。
    """
    logger.info(f"Using device: {device}")
    device = torch.device(device)
    
    # 加载数据
    logger.info(f"Loading train data from {train_path}")
    train_df = pd.read_parquet(train_path)
    train_dataset = LNPDataset(train_df)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    
    logger.info(f"Loading val data from {val_path}")
    val_df = pd.read_parquet(val_path)
    val_dataset = LNPDataset(val_df)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 超参数调优
    if tune:
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")
        best_params = run_hyperparameter_tuning(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            n_trials=n_trials,
            epochs_per_trial=epochs_per_trial,
        )
        
        # 保存最佳参数
        params_path = output_dir / "best_params.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        logger.success(f"Saved best params to {params_path}")
        
        # 使用最佳参数重新训练
        d_model = best_params["d_model"]
        num_heads = best_params["num_heads"]
        n_attn_layers = best_params["n_attn_layers"]
        fusion_strategy = best_params["fusion_strategy"]
        head_hidden_dim = best_params["head_hidden_dim"]
        dropout = best_params["dropout"]
        lr = best_params["lr"]
        weight_decay = best_params["weight_decay"]
    
    # 创建模型
    logger.info("Creating model...")
    model = create_model(
        d_model=d_model,
        num_heads=num_heads,
        n_attn_layers=n_attn_layers,
        fusion_strategy=fusion_strategy,
        head_hidden_dim=head_hidden_dim,
        dropout=dropout,
    )
    
    # 打印模型信息
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    
    # 训练
    logger.info("Starting training...")
    result = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=patience,
    )
    
    # 保存模型
    model_path = output_dir / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "d_model": d_model,
            "num_heads": num_heads,
            "n_attn_layers": n_attn_layers,
            "fusion_strategy": fusion_strategy,
            "head_hidden_dim": head_hidden_dim,
            "dropout": dropout,
        },
        "best_val_loss": result["best_val_loss"],
    }, model_path)
    logger.success(f"Saved model to {model_path}")
    
    # 保存训练历史
    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(result["history"], f, indent=2)
    logger.success(f"Saved training history to {history_path}")
    
    logger.success(f"Training complete! Best val_loss: {result['best_val_loss']:.4f}")


if __name__ == "__main__":
    app()
