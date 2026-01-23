"""训练脚本：支持超参数调优"""

import json
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader
from loguru import logger
import typer

from lnp_ml.config import MODELS_DIR, PROCESSED_DATA_DIR
from lnp_ml.dataset import LNPDataset, collate_fn
from lnp_ml.modeling.models import LNPModel, LNPModelWithoutMPNN
from lnp_ml.modeling.trainer import (
    train_epoch,
    validate,
    EarlyStopping,
    LossWeights,
)
from lnp_ml.modeling.visualization import plot_multitask_loss_curves

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


app = typer.Typer()


def create_model(
    d_model: int = 256,
    num_heads: int = 8,
    n_attn_layers: int = 4,
    fusion_strategy: str = "attention",
    head_hidden_dim: int = 128,
    dropout: float = 0.1,
    # MPNN 参数（可选）
    mpnn_checkpoint: Optional[str] = None,
    mpnn_ensemble_paths: Optional[List[str]] = None,
    mpnn_device: str = "cpu",
) -> Union[LNPModel, LNPModelWithoutMPNN]:
    """创建模型（支持可选的 MPNN encoder）"""
    use_mpnn = mpnn_checkpoint is not None or mpnn_ensemble_paths is not None
    
    if use_mpnn:
        return LNPModel(
            d_model=d_model,
            num_heads=num_heads,
            n_attn_layers=n_attn_layers,
            fusion_strategy=fusion_strategy,
            head_hidden_dim=head_hidden_dim,
            dropout=dropout,
            mpnn_checkpoint=mpnn_checkpoint,
            mpnn_ensemble_paths=mpnn_ensemble_paths,
            mpnn_device=mpnn_device,
        )
    else:
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
    # MPNN 参数（可选）
    use_mpnn: bool = False,  # 启用 MPNN，自动从默认路径加载 ensemble
    mpnn_checkpoint: Optional[str] = None,
    mpnn_ensemble_paths: Optional[str] = None,  # 逗号分隔的路径列表
    mpnn_device: str = "cpu",
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
    # 预训练权重加载
    init_from_pretrain: Optional[Path] = None,
    load_delivery_head: bool = True,
    freeze_backbone: bool = False,  # 冻结 backbone，只训练 heads
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    训练 LNP 预测模型（多任务 finetune）。
    
    使用 --tune 启用超参数调优。
    使用 --init-from-pretrain 从预训练 checkpoint 初始化 backbone。
    使用 --use-mpnn 启用 MPNN encoder（自动从 models/mpnn/all_amine_split_for_LiON 加载）。
    使用 --freeze-backbone 冻结 backbone，只训练多任务 heads。
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
    model = create_model(
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
    
    # 加载预训练权重（如果指定）
    if init_from_pretrain is not None:
        logger.info(f"Loading pretrain weights from {init_from_pretrain}")
        checkpoint = torch.load(init_from_pretrain, map_location="cpu")
        
        # 检查配置是否兼容
        pretrain_config = checkpoint.get("config", {})
        if pretrain_config.get("d_model") != d_model:
            logger.warning(
                f"d_model mismatch: pretrain={pretrain_config.get('d_model')}, "
                f"current={d_model}. Skipping pretrain loading."
            )
        else:
            # 加载 backbone + (可选) delivery head
            model.load_pretrain_weights(
                pretrain_state_dict=checkpoint["model_state_dict"],
                load_delivery_head=load_delivery_head,
                strict=False,
            )
            logger.success(
                f"Loaded pretrain weights (backbone + delivery_head={load_delivery_head})"
            )
    
    # 冻结 backbone（如果指定）
    if freeze_backbone:
        logger.info("Freezing backbone (token_projector, cross_attention, fusion)...")
        frozen_count = 0
        for name, param in model.named_parameters():
            if name.startswith(("token_projector.", "cross_attention.", "fusion.")):
                param.requires_grad = False
                frozen_count += 1
        logger.info(f"Frozen {frozen_count} parameter tensors")
    
    # 打印模型信息
    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params_total:,} total, {n_params_trainable:,} trainable")
    
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
            "use_mpnn": enable_mpnn,
        },
        "best_val_loss": result["best_val_loss"],
        "init_from_pretrain": str(init_from_pretrain) if init_from_pretrain else None,
    }, model_path)
    logger.success(f"Saved model to {model_path}")
    
    # 保存训练历史
    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(result["history"], f, indent=2)
    logger.success(f"Saved training history to {history_path}")
    
    # 绘制多任务 loss 曲线图
    loss_plot_path = output_dir / "loss_curves.png"
    plot_multitask_loss_curves(
        history=result["history"],
        output_path=loss_plot_path,
        title="Multi-task Training Loss Curves",
    )
    logger.success(f"Saved loss curves plot to {loss_plot_path}")
    
    logger.success(f"Training complete! Best val_loss: {result['best_val_loss']:.4f}")


if __name__ == "__main__":
    app()
