"""Cross-Validation 训练脚本：在 5-fold 内部数据上进行多任务训练"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm
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
    """自动查找 MPNN ensemble 的 model.pt 文件。"""
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


def train_fold(
    fold_idx: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    output_dir: Path,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    epochs: int = 100,
    patience: int = 15,
    loss_weights: Optional[LossWeights] = None,
    config: Optional[Dict] = None,
) -> Dict:
    """训练单个 fold"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Fold {fold_idx}")
    logger.info(f"{'='*60}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
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
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Log
        logger.info(
            f"Fold {fold_idx} Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"LR: {current_lr:.2e}"
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
    
    # 保存最佳模型
    fold_output_dir = output_dir / f"fold_{fold_idx}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = fold_output_dir / "model.pt"
    torch.save({
        "model_state_dict": best_state,
        "config": config,
        "best_val_loss": best_val_loss,
        "fold_idx": fold_idx,
    }, checkpoint_path)
    logger.success(f"Saved fold {fold_idx} model to {checkpoint_path}")
    
    # 保存训练历史
    history_path = fold_output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    # 绘制多任务 loss 曲线图
    loss_plot_path = fold_output_dir / "loss_curves.png"
    plot_multitask_loss_curves(
        history=history,
        output_path=loss_plot_path,
        title=f"Fold {fold_idx} Multi-task Loss Curves",
    )
    logger.info(f"Saved fold {fold_idx} loss curves to {loss_plot_path}")
    
    return {
        "fold_idx": fold_idx,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["train"]),
        "final_train_loss": history["train"][-1]["loss"] if history["train"] else 0,
    }


@app.command()
def main(
    data_dir: Path = PROCESSED_DATA_DIR / "cv",
    output_dir: Path = MODELS_DIR / "finetune_cv",
    # 模型参数
    d_model: int = 256,
    num_heads: int = 8,
    n_attn_layers: int = 4,
    fusion_strategy: str = "attention",
    head_hidden_dim: int = 128,
    dropout: float = 0.1,
    # MPNN 参数（可选）
    use_mpnn: bool = False,
    mpnn_checkpoint: Optional[str] = None,
    mpnn_ensemble_paths: Optional[str] = None,
    mpnn_device: str = "cpu",
    # 训练参数
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    epochs: int = 100,
    patience: int = 15,
    # 预训练权重加载
    init_from_pretrain: Optional[Path] = None,
    load_delivery_head: bool = True,
    freeze_backbone: bool = False,
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    基于 Cross-Validation 训练 LNP 模型（多任务）。
    
    在 5-fold 内部数据上训练 5 个模型。
    
    使用 --use-mpnn 启用 MPNN encoder。
    使用 --init-from-pretrain 从预训练 checkpoint 初始化。
    使用 --freeze-backbone 冻结 backbone，只训练 heads。
    """
    logger.info(f"Using device: {device}")
    device = torch.device(device)
    
    # 查找所有 fold 目录
    fold_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    
    if not fold_dirs:
        logger.error(f"No fold_* directories found in {data_dir}")
        logger.info("Please run 'make data_cv' first to process CV data.")
        raise typer.Exit(1)
    
    logger.info(f"Found {len(fold_dirs)} folds: {[d.name for d in fold_dirs]}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析 MPNN 配置
    ensemble_paths_list = None
    if mpnn_ensemble_paths:
        ensemble_paths_list = mpnn_ensemble_paths.split(",")
    elif use_mpnn and mpnn_checkpoint is None:
        logger.info(f"Auto-detecting MPNN ensemble from {DEFAULT_MPNN_ENSEMBLE_DIR}")
        ensemble_paths_list = find_mpnn_ensemble_paths()
        logger.info(f"Found {len(ensemble_paths_list)} MPNN models")
    
    enable_mpnn = mpnn_checkpoint is not None or ensemble_paths_list is not None
    
    # 模型配置
    config = {
        "d_model": d_model,
        "num_heads": num_heads,
        "n_attn_layers": n_attn_layers,
        "fusion_strategy": fusion_strategy,
        "head_hidden_dim": head_hidden_dim,
        "dropout": dropout,
        "use_mpnn": enable_mpnn,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "init_from_pretrain": str(init_from_pretrain) if init_from_pretrain else None,
        "freeze_backbone": freeze_backbone,
    }
    
    # 保存配置
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_path}")
    
    # 加载预训练权重（如果指定）
    pretrain_state = None
    if init_from_pretrain is not None:
        logger.info(f"Loading pretrain weights from {init_from_pretrain}")
        checkpoint = torch.load(init_from_pretrain, map_location="cpu")
        pretrain_config = checkpoint.get("config", {})
        if pretrain_config.get("d_model") != d_model:
            logger.warning(
                f"d_model mismatch: pretrain={pretrain_config.get('d_model')}, "
                f"current={d_model}. Skipping pretrain loading."
            )
        else:
            pretrain_state = checkpoint["model_state_dict"]
    
    # 训练每个 fold
    fold_results = []
    
    for fold_dir in tqdm(fold_dirs, desc="Training folds"):
        fold_idx = int(fold_dir.name.split("_")[1])
        
        # 加载数据
        train_df = pd.read_parquet(fold_dir / "train.parquet")
        val_df = pd.read_parquet(fold_dir / "val.parquet")
        
        logger.info(f"\nFold {fold_idx}: train={len(train_df)}, val={len(val_df)}")
        
        # 创建 Dataset 和 DataLoader
        train_dataset = LNPDataset(train_df)
        val_dataset = LNPDataset(val_df)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        
        # 创建新模型（每个 fold 独立初始化）
        model = create_model(
            d_model=d_model,
            num_heads=num_heads,
            n_attn_layers=n_attn_layers,
            fusion_strategy=fusion_strategy,
            head_hidden_dim=head_hidden_dim,
            dropout=dropout,
            mpnn_checkpoint=mpnn_checkpoint,
            mpnn_ensemble_paths=ensemble_paths_list,
            mpnn_device=device.type,
        )
        
        # 加载预训练权重
        if pretrain_state is not None:
            model.load_pretrain_weights(
                pretrain_state_dict=pretrain_state,
                load_delivery_head=load_delivery_head,
                strict=False,
            )
            logger.info(f"Loaded pretrain weights (backbone + delivery_head={load_delivery_head})")
        
        # 冻结 backbone（如果指定）
        if freeze_backbone:
            frozen_count = 0
            for name, param in model.named_parameters():
                if name.startswith(("token_projector.", "cross_attention.", "fusion.")):
                    param.requires_grad = False
                    frozen_count += 1
            logger.info(f"Frozen {frozen_count} parameter tensors")
        
        # 打印模型信息（仅第一个 fold）
        if fold_idx == 0:
            n_params_total = sum(p.numel() for p in model.parameters())
            n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model parameters: {n_params_total:,} total, {n_params_trainable:,} trainable")
        
        # 训练
        result = train_fold(
            fold_idx=fold_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            device=device,
            output_dir=output_dir,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            patience=patience,
            config=config,
        )
        fold_results.append(result)
    
    # 汇总结果
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-VALIDATION TRAINING COMPLETE")
    logger.info("=" * 60)
    
    val_losses = [r["best_val_loss"] for r in fold_results]
    
    logger.info(f"\n[Per-Fold Results]")
    for r in fold_results:
        logger.info(
            f"  Fold {r['fold_idx']}: "
            f"Val Loss={r['best_val_loss']:.4f}, "
            f"Epochs={r['epochs_trained']}"
        )
    
    logger.info(f"\n[Summary Statistics]")
    logger.info(f"  Val Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    
    # 保存 CV 结果
    cv_results = {
        "fold_results": fold_results,
        "summary": {
            "val_loss_mean": float(np.mean(val_losses)),
            "val_loss_std": float(np.std(val_losses)),
        },
        "config": config,
    }
    
    results_path = output_dir / "cv_results.json"
    with open(results_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    logger.success(f"Saved CV results to {results_path}")


@app.command()
def test(
    data_dir: Path = PROCESSED_DATA_DIR / "cv",
    model_dir: Path = MODELS_DIR / "finetune_cv",
    output_path: Path = MODELS_DIR / "finetune_cv" / "test_results.json",
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    在测试集上评估 CV 训练的模型。
    
    使用每个 fold 的模型在对应的测试集上评估，然后汇总结果。
    """
    from scipy.special import rel_entr
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )
    
    def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
        """计算 KL 散度 KL(p || q)"""
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        return float(np.sum(rel_entr(p, q), axis=-1).mean())
    
    def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
        """计算 JS 散度"""
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        m = 0.5 * (p + q)
        return float(0.5 * (np.sum(rel_entr(p, m), axis=-1) + np.sum(rel_entr(q, m), axis=-1)).mean())
    
    logger.info(f"Using device: {device}")
    device = torch.device(device)
    
    # 查找所有 fold 目录
    fold_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    
    if not fold_dirs:
        logger.error(f"No fold_* directories found in {data_dir}")
        raise typer.Exit(1)
    
    logger.info(f"Found {len(fold_dirs)} folds")
    
    fold_results = []
    # 用于汇总所有 fold 的预测
    all_preds = {
        "size": [], "delivery": [], "pdi": [], "ee": [], "toxic": [], "biodist": []
    }
    all_targets = {
        "size": [], "delivery": [], "pdi": [], "ee": [], "toxic": [], "biodist": []
    }
    
    for fold_dir in tqdm(fold_dirs, desc="Evaluating folds"):
        fold_idx = int(fold_dir.name.split("_")[1])
        model_path = model_dir / f"fold_{fold_idx}" / "model.pt"
        test_path = fold_dir / "test.parquet"
        
        if not model_path.exists():
            logger.warning(f"Fold {fold_idx}: model not found at {model_path}, skipping")
            continue
        
        if not test_path.exists():
            logger.warning(f"Fold {fold_idx}: test data not found at {test_path}, skipping")
            continue
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint["config"]
        
        use_mpnn = config.get("use_mpnn", False)
        
        # 总是重新查找 MPNN 路径
        if use_mpnn:
            mpnn_paths = find_mpnn_ensemble_paths()
        else:
            mpnn_paths = None
        
        model = create_model(
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            n_attn_layers=config["n_attn_layers"],
            fusion_strategy=config["fusion_strategy"],
            head_hidden_dim=config["head_hidden_dim"],
            dropout=config["dropout"],
            mpnn_ensemble_paths=mpnn_paths,
            mpnn_device=device.type,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        
        # 加载测试数据
        test_df = pd.read_parquet(test_path)
        test_dataset = LNPDataset(test_df)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        
        # 收集当前 fold 的预测
        fold_preds = {k: [] for k in all_preds.keys()}
        fold_targets = {k: [] for k in all_targets.keys()}
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Fold {fold_idx} [Test]", leave=False)
            for batch in pbar:
                smiles = batch["smiles"]
                tabular = {k: v.to(device) for k, v in batch["tabular"].items()}
                targets = batch["targets"]
                masks = batch["mask"]
                
                outputs = model(smiles, tabular)
                
                # Size
                if "size" in masks and masks["size"].any():
                    mask = masks["size"]
                    fold_preds["size"].extend(
                        outputs["size"].squeeze(-1)[mask].cpu().numpy().tolist()
                    )
                    fold_targets["size"].extend(
                        targets["size"][mask].cpu().numpy().tolist()
                    )
                
                # Delivery
                if "delivery" in masks and masks["delivery"].any():
                    mask = masks["delivery"]
                    fold_preds["delivery"].extend(
                        outputs["delivery"].squeeze(-1)[mask].cpu().numpy().tolist()
                    )
                    fold_targets["delivery"].extend(
                        targets["delivery"][mask].cpu().numpy().tolist()
                    )
                
                # PDI (classification)
                if "pdi" in masks and masks["pdi"].any():
                    mask = masks["pdi"]
                    pdi_preds = outputs["pdi"][mask].argmax(dim=-1).cpu().numpy()
                    pdi_targets = targets["pdi"][mask].cpu().numpy()
                    fold_preds["pdi"].extend(pdi_preds.tolist())
                    fold_targets["pdi"].extend(pdi_targets.tolist())
                
                # EE (classification)
                if "ee" in masks and masks["ee"].any():
                    mask = masks["ee"]
                    ee_preds = outputs["ee"][mask].argmax(dim=-1).cpu().numpy()
                    ee_targets = targets["ee"][mask].cpu().numpy()
                    fold_preds["ee"].extend(ee_preds.tolist())
                    fold_targets["ee"].extend(ee_targets.tolist())
                
                # Toxic (classification)
                if "toxic" in masks and masks["toxic"].any():
                    mask = masks["toxic"]
                    toxic_preds = outputs["toxic"][mask].argmax(dim=-1).cpu().numpy()
                    toxic_targets = targets["toxic"][mask].cpu().numpy().astype(int)
                    fold_preds["toxic"].extend(toxic_preds.tolist())
                    fold_targets["toxic"].extend(toxic_targets.tolist())
                
                # Biodist (distribution)
                if "biodist" in masks and masks["biodist"].any():
                    mask = masks["biodist"]
                    biodist_preds = outputs["biodist"][mask].cpu().numpy()
                    biodist_targets = targets["biodist"][mask].cpu().numpy()
                    fold_preds["biodist"].extend(biodist_preds.tolist())
                    fold_targets["biodist"].extend(biodist_targets.tolist())
        
        # 计算当前 fold 的指标
        fold_metrics = {"fold_idx": fold_idx, "n_samples": len(test_df)}
        
        # 回归任务指标
        for task in ["size", "delivery"]:
            if fold_preds[task]:
                p = np.array(fold_preds[task])
                t = np.array(fold_targets[task])
                fold_metrics[task] = {
                    "n": len(p),
                    "rmse": float(np.sqrt(mean_squared_error(t, p))),
                    "mae": float(mean_absolute_error(t, p)),
                    "r2": float(r2_score(t, p)),
                }
        
        # 分类任务指标
        for task in ["pdi", "ee", "toxic"]:
            if fold_preds[task]:
                p = np.array(fold_preds[task])
                t = np.array(fold_targets[task])
                fold_metrics[task] = {
                    "n": len(p),
                    "accuracy": float(accuracy_score(t, p)),
                    "precision": float(precision_score(t, p, average="macro", zero_division=0)),
                    "recall": float(recall_score(t, p, average="macro", zero_division=0)),
                    "f1": float(f1_score(t, p, average="macro", zero_division=0)),
                }
        
        # 分布任务指标
        if fold_preds["biodist"]:
            p = np.array(fold_preds["biodist"])
            t = np.array(fold_targets["biodist"])
            fold_metrics["biodist"] = {
                "n": len(p),
                "kl_divergence": kl_divergence(t, p),
                "js_divergence": js_divergence(t, p),
            }
        
        fold_results.append(fold_metrics)
        
        # 汇总到全局
        for task in all_preds.keys():
            all_preds[task].extend(fold_preds[task])
            all_targets[task].extend(fold_targets[task])
        
        # 打印当前 fold 结果
        log_parts = [f"Fold {fold_idx}: n={len(test_df)}"]
        for task in ["delivery", "size"]:
            if task in fold_metrics and isinstance(fold_metrics[task], dict):
                log_parts.append(f"{task}_RMSE={fold_metrics[task]['rmse']:.4f}")
                log_parts.append(f"{task}_R²={fold_metrics[task]['r2']:.4f}")
        for task in ["pdi", "ee", "toxic"]:
            if task in fold_metrics and isinstance(fold_metrics[task], dict):
                log_parts.append(f"{task}_acc={fold_metrics[task]['accuracy']:.4f}")
                log_parts.append(f"{task}_f1={fold_metrics[task]['f1']:.4f}")
        if "biodist" in fold_metrics and isinstance(fold_metrics["biodist"], dict):
            log_parts.append(f"biodist_KL={fold_metrics['biodist']['kl_divergence']:.4f}")
            log_parts.append(f"biodist_JS={fold_metrics['biodist']['js_divergence']:.4f}")
        logger.info(", ".join(log_parts))
    
    # 计算跨 fold 汇总统计
    summary_stats = {}
    for task in ["size", "delivery"]:
        rmses = [r[task]["rmse"] for r in fold_results if task in r and isinstance(r[task], dict)]
        r2s = [r[task]["r2"] for r in fold_results if task in r and isinstance(r[task], dict)]
        if rmses:
            summary_stats[task] = {
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
                "r2_mean": float(np.mean(r2s)),
                "r2_std": float(np.std(r2s)),
            }
    
    for task in ["pdi", "ee", "toxic"]:
        accs = [r[task]["accuracy"] for r in fold_results if task in r and isinstance(r[task], dict)]
        f1s = [r[task]["f1"] for r in fold_results if task in r and isinstance(r[task], dict)]
        if accs:
            summary_stats[task] = {
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s)),
            }
    
    # 分布任务汇总
    kls = [r["biodist"]["kl_divergence"] for r in fold_results if "biodist" in r and isinstance(r["biodist"], dict)]
    jss = [r["biodist"]["js_divergence"] for r in fold_results if "biodist" in r and isinstance(r["biodist"], dict)]
    if kls:
        summary_stats["biodist"] = {
            "kl_mean": float(np.mean(kls)),
            "kl_std": float(np.std(kls)),
            "js_mean": float(np.mean(jss)),
            "js_std": float(np.std(jss)),
        }
    
    # 计算整体 pooled 指标
    overall = {}
    for task in ["size", "delivery"]:
        if all_preds[task]:
            p = np.array(all_preds[task])
            t = np.array(all_targets[task])
            overall[task] = {
                "n_samples": len(p),
                "mse": float(mean_squared_error(t, p)),
                "rmse": float(np.sqrt(mean_squared_error(t, p))),
                "mae": float(mean_absolute_error(t, p)),
                "r2": float(r2_score(t, p)),
            }
    
    for task in ["pdi", "ee", "toxic"]:
        if all_preds[task]:
            p = np.array(all_preds[task])
            t = np.array(all_targets[task])
            overall[task] = {
                "n_samples": len(p),
                "accuracy": float(accuracy_score(t, p)),
                "precision": float(precision_score(t, p, average="macro", zero_division=0)),
                "recall": float(recall_score(t, p, average="macro", zero_division=0)),
                "f1": float(f1_score(t, p, average="macro", zero_division=0)),
            }
    
    # 分布任务
    if all_preds["biodist"]:
        p = np.array(all_preds["biodist"])
        t = np.array(all_targets["biodist"])
        overall["biodist"] = {
            "n_samples": len(p),
            "kl_divergence": kl_divergence(t, p),
            "js_divergence": js_divergence(t, p),
        }
    
    # 打印汇总结果
    logger.info("\n" + "=" * 60)
    logger.info("CV TEST EVALUATION RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"\n[Summary Statistics (across {len(fold_results)} folds)]")
    for task, stats in summary_stats.items():
        if "rmse_mean" in stats:
            logger.info(f"  {task}: RMSE={stats['rmse_mean']:.4f}±{stats['rmse_std']:.4f}, R²={stats['r2_mean']:.4f}±{stats['r2_std']:.4f}")
        elif "accuracy_mean" in stats:
            logger.info(f"  {task}: Accuracy={stats['accuracy_mean']:.4f}±{stats['accuracy_std']:.4f}, F1={stats['f1_mean']:.4f}±{stats['f1_std']:.4f}")
        elif "kl_mean" in stats:
            logger.info(f"  {task}: KL={stats['kl_mean']:.4f}±{stats['kl_std']:.4f}, JS={stats['js_mean']:.4f}±{stats['js_std']:.4f}")
    
    logger.info(f"\n[Overall (all samples pooled)]")
    for task, metrics in overall.items():
        if "rmse" in metrics:
            logger.info(f"  {task} (n={metrics['n_samples']}): RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")
        elif "accuracy" in metrics:
            logger.info(f"  {task} (n={metrics['n_samples']}): Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        elif "kl_divergence" in metrics:
            logger.info(f"  {task} (n={metrics['n_samples']}): KL={metrics['kl_divergence']:.4f}, JS={metrics['js_divergence']:.4f}")
    
    # 保存结果
    results = {
        "fold_results": fold_results,
        "summary_stats": summary_stats,
        "overall": overall,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"\nSaved test results to {output_path}")


if __name__ == "__main__":
    app()

