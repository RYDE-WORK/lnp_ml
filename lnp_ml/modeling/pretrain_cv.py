"""基于 Cross-Validation 的预训练脚本"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
import typer

from lnp_ml.config import MODELS_DIR, PROCESSED_DATA_DIR
from lnp_ml.dataset import ExternalDeliveryDataset, collate_fn


# MPNN ensemble 默认路径
DEFAULT_MPNN_ENSEMBLE_DIR = MODELS_DIR / "mpnn" / "all_amine_split_for_LiON"


def find_mpnn_ensemble_paths(base_dir: Path = DEFAULT_MPNN_ENSEMBLE_DIR) -> List[str]:
    """
    自动查找 MPNN ensemble 的 model.pt 文件。
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
    """预热 RDKit 特征缓存"""
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
    """单个 epoch 的训练（仅 delivery 任务）"""
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

        pred = model.forward_delivery(smiles, tabular).squeeze(-1)

        if mask.any():
            loss = nn.functional.mse_loss(pred[mask], targets[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    """验证（仅 delivery 任务）"""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_preds = []
    all_targets = []

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
            all_preds.extend(pred[mask].cpu().numpy().tolist())
            all_targets.extend(targets[mask].cpu().numpy().tolist())

    avg_loss = total_loss / max(n_samples, 1)
    
    # 计算额外指标
    metrics = {"loss": avg_loss, "n_samples": n_samples}
    if len(all_preds) > 0:
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        metrics["rmse"] = float(np.sqrt(mean_squared_error(all_targets, all_preds)))
        metrics["r2"] = float(r2_score(all_targets, all_preds))
    
    return metrics


def train_fold(
    fold_idx: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    output_dir: Path,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    epochs: int = 50,
    patience: int = 10,
    config: Optional[Dict] = None,
) -> Dict:
    """训练单个 fold"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Fold {fold_idx}")
    logger.info(f"{'='*60}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(patience=patience)
    
    best_val_loss = float("inf")
    best_state = None
    history = []
    
    for epoch in range(epochs):
        train_metrics = train_epoch_delivery(model, train_loader, optimizer, device, epoch)
        val_metrics = validate_delivery(model, val_loader, device)
        
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Fold {fold_idx} Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val RMSE: {val_metrics.get('rmse', 0):.4f} | "
            f"Val R²: {val_metrics.get('r2', 0):.4f} | "
            f"LR: {current_lr:.2e}"
        )
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_rmse": val_metrics.get("rmse", 0),
            "val_r2": val_metrics.get("r2", 0),
            "lr": current_lr,
        })
        
        scheduler.step(val_metrics["loss"])
        
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  -> New best val_loss: {best_val_loss:.4f}")
        
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
    
    return {
        "fold_idx": fold_idx,
        "best_val_loss": best_val_loss,
        "best_val_rmse": history[-1]["val_rmse"] if history else 0,
        "best_val_r2": history[-1]["val_r2"] if history else 0,
        "epochs_trained": len(history),
    }


def create_model(
    d_model: int = 256,
    num_heads: int = 8,
    n_attn_layers: int = 4,
    fusion_strategy: str = "attention",
    head_hidden_dim: int = 128,
    dropout: float = 0.1,
    use_mpnn: bool = False,
    mpnn_ensemble_paths: Optional[List[str]] = None,
    mpnn_device: str = "cpu",
) -> nn.Module:
    """创建模型实例"""
    if use_mpnn:
        return LNPModel(
            d_model=d_model,
            num_heads=num_heads,
            n_attn_layers=n_attn_layers,
            fusion_strategy=fusion_strategy,
            head_hidden_dim=head_hidden_dim,
            dropout=dropout,
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


@app.command()
def main(
    data_dir: Path = PROCESSED_DATA_DIR / "cv",
    output_dir: Path = MODELS_DIR / "pretrain_cv",
    # 模型参数
    d_model: int = 256,
    num_heads: int = 8,
    n_attn_layers: int = 4,
    fusion_strategy: str = "attention",
    head_hidden_dim: int = 128,
    dropout: float = 0.1,
    # MPNN 参数
    use_mpnn: bool = False,
    mpnn_checkpoint: Optional[str] = None,
    mpnn_ensemble_paths: Optional[str] = None,
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
    基于 5-fold Cross-Validation 预训练 LNP 模型（仅 delivery 任务）。
    
    每个 fold 单独训练一个模型，保存到 output_dir/fold_x/model.pt。
    使用 --use-mpnn 启用 MPNN encoder。
    """
    logger.info(f"Using device: {device}")
    device = torch.device(device)
    
    # 解析 MPNN 参数
    mpnn_paths = None
    if use_mpnn:
        if mpnn_ensemble_paths:
            mpnn_paths = mpnn_ensemble_paths.split(",")
            logger.info(f"Using provided MPNN ensemble paths: {len(mpnn_paths)} models")
        elif mpnn_checkpoint:
            mpnn_paths = [mpnn_checkpoint]
            logger.info(f"Using single MPNN checkpoint: {mpnn_checkpoint}")
        else:
            logger.info(f"Auto-detecting MPNN ensemble from {DEFAULT_MPNN_ENSEMBLE_DIR}")
            mpnn_paths = find_mpnn_ensemble_paths()
            logger.info(f"Found {len(mpnn_paths)} MPNN models")
    
    # 查找所有 fold 目录
    fold_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    
    if not fold_dirs:
        logger.error(f"No fold_* directories found in {data_dir}")
        logger.info("Please run 'make data_cv' first to process CV data.")
        raise typer.Exit(1)
    
    logger.info(f"Found {len(fold_dirs)} folds: {[d.name for d in fold_dirs]}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 模型配置
    config = {
        "d_model": d_model,
        "num_heads": num_heads,
        "n_attn_layers": n_attn_layers,
        "fusion_strategy": fusion_strategy,
        "head_hidden_dim": head_hidden_dim,
        "dropout": dropout,
        "use_mpnn": use_mpnn,
        "mpnn_ensemble_paths": mpnn_paths,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
    }
    
    # 保存配置
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_path}")
    
    # 训练每个 fold
    fold_results = []
    all_smiles = set()
    
    # 先收集所有 SMILES 用于 cache warmup
    for fold_dir in fold_dirs:
        for split in ["train", "valid"]:
            df = pd.read_parquet(fold_dir / f"{split}.parquet")
            all_smiles.update(df["smiles"].tolist())
    
    for fold_dir in fold_dirs:
        fold_idx = int(fold_dir.name.split("_")[1])
        
        # 加载数据
        train_df = pd.read_parquet(fold_dir / "train.parquet")
        val_df = pd.read_parquet(fold_dir / "valid.parquet")
        
        logger.info(f"\nFold {fold_idx}: train={len(train_df)}, val={len(val_df)}")
        
        # 创建 Dataset 和 DataLoader
        train_dataset = ExternalDeliveryDataset(train_df)
        val_dataset = ExternalDeliveryDataset(val_df)
        
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
            use_mpnn=use_mpnn,
            mpnn_ensemble_paths=mpnn_paths,
            mpnn_device=device.type,
        )
        model = model.to(device)
        
        # 第一个 fold 时做 cache warmup
        if fold_idx == 0:
            warmup_cache(model, list(all_smiles), batch_size=256)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
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
    val_rmses = [r["best_val_rmse"] for r in fold_results]
    val_r2s = [r["best_val_r2"] for r in fold_results]
    
    logger.info(f"\n[Per-Fold Results]")
    for r in fold_results:
        logger.info(
            f"  Fold {r['fold_idx']}: "
            f"Val Loss={r['best_val_loss']:.4f}, "
            f"RMSE={r['best_val_rmse']:.4f}, "
            f"R²={r['best_val_r2']:.4f}, "
            f"Epochs={r['epochs_trained']}"
        )
    
    logger.info(f"\n[Summary Statistics]")
    logger.info(f"  Val Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    logger.info(f"  Val RMSE: {np.mean(val_rmses):.4f} ± {np.std(val_rmses):.4f}")
    logger.info(f"  Val R²:   {np.mean(val_r2s):.4f} ± {np.std(val_r2s):.4f}")
    
    # 保存 CV 结果
    cv_results = {
        "fold_results": fold_results,
        "summary": {
            "val_loss_mean": float(np.mean(val_losses)),
            "val_loss_std": float(np.std(val_losses)),
            "val_rmse_mean": float(np.mean(val_rmses)),
            "val_rmse_std": float(np.std(val_rmses)),
            "val_r2_mean": float(np.mean(val_r2s)),
            "val_r2_std": float(np.std(val_r2s)),
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
    model_dir: Path = MODELS_DIR / "pretrain_cv",
    output_path: Path = MODELS_DIR / "pretrain_cv" / "test_results.json",
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    在测试集上评估 CV 预训练模型。
    
    使用每个 fold 的模型在对应的测试集上评估。
    """
    logger.info(f"Using device: {device}")
    device = torch.device(device)
    
    # 查找所有 fold 目录
    fold_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    
    if not fold_dirs:
        logger.error(f"No fold_* directories found in {data_dir}")
        raise typer.Exit(1)
    
    logger.info(f"Found {len(fold_dirs)} folds")
    
    fold_results = []
    all_preds = []
    all_targets = []
    
    for fold_dir in fold_dirs:
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
        mpnn_paths = config.get("mpnn_ensemble_paths")
        
        if use_mpnn and not mpnn_paths:
            mpnn_paths = find_mpnn_ensemble_paths()
        
        model = create_model(
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            n_attn_layers=config["n_attn_layers"],
            fusion_strategy=config["fusion_strategy"],
            head_hidden_dim=config["head_hidden_dim"],
            dropout=config["dropout"],
            use_mpnn=use_mpnn,
            mpnn_ensemble_paths=mpnn_paths,
            mpnn_device=device.type,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        
        # 加载测试数据
        test_df = pd.read_parquet(test_path)
        test_dataset = ExternalDeliveryDataset(test_df)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        
        # 评估
        fold_preds = []
        fold_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                smiles = batch["smiles"]
                tabular = {k: v.to(device) for k, v in batch["tabular"].items()}
                targets = batch["targets"]["delivery"].to(device)
                mask = batch["mask"]["delivery"].to(device)
                
                pred = model.forward_delivery(smiles, tabular).squeeze(-1)
                
                if mask.any():
                    fold_preds.extend(pred[mask].cpu().numpy().tolist())
                    fold_targets.extend(targets[mask].cpu().numpy().tolist())
        
        # 计算 fold 指标
        fold_preds = np.array(fold_preds)
        fold_targets = np.array(fold_targets)
        
        mse = float(mean_squared_error(fold_targets, fold_preds))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(fold_targets, fold_preds))
        mae = float(np.mean(np.abs(fold_targets - fold_preds)))
        corr = float(np.corrcoef(fold_targets, fold_preds)[0, 1])
        
        fold_results.append({
            "fold_idx": fold_idx,
            "n_samples": len(fold_preds),
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "correlation": corr,
        })
        
        all_preds.extend(fold_preds.tolist())
        all_targets.extend(fold_targets.tolist())
        
        logger.info(
            f"Fold {fold_idx}: n={len(fold_preds)}, "
            f"RMSE={rmse:.4f}, R²={r2:.4f}, MAE={mae:.4f}, Corr={corr:.4f}"
        )
    
    # 计算整体指标
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    overall_mse = float(mean_squared_error(all_targets, all_preds))
    overall_rmse = float(np.sqrt(overall_mse))
    overall_r2 = float(r2_score(all_targets, all_preds))
    overall_mae = float(np.mean(np.abs(all_targets - all_preds)))
    overall_corr = float(np.corrcoef(all_targets, all_preds)[0, 1])
    
    # 汇总统计
    rmses = [r["rmse"] for r in fold_results]
    r2s = [r["r2"] for r in fold_results]
    
    logger.info("\n" + "=" * 60)
    logger.info("CV TEST EVALUATION RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"\n[Summary Statistics (across {len(fold_results)} folds)]")
    logger.info(f"  RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    logger.info(f"  R²:   {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
    
    logger.info(f"\n[Overall (all {len(all_preds)} samples pooled)]")
    logger.info(f"  RMSE:        {overall_rmse:.4f}")
    logger.info(f"  R²:          {overall_r2:.4f}")
    logger.info(f"  MAE:         {overall_mae:.4f}")
    logger.info(f"  Correlation: {overall_corr:.4f}")
    
    # 保存结果
    results = {
        "fold_results": fold_results,
        "summary_stats": {
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
        },
        "overall": {
            "n_samples": len(all_preds),
            "mse": overall_mse,
            "rmse": overall_rmse,
            "mae": overall_mae,
            "r2": overall_r2,
            "correlation": overall_corr,
        },
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"Saved test results to {output_path}")


if __name__ == "__main__":
    app()

