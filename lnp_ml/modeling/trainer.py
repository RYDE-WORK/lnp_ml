"""训练器：封装训练、验证、损失计算逻辑"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class LossWeights:
    """各任务的损失权重"""
    size: float = 1.0
    pdi: float = 1.0
    ee: float = 1.0
    delivery: float = 1.0
    biodist: float = 1.0
    toxic: float = 1.0
    # size: float = 0.1
    # pdi: float = 0.3
    # ee: float = 0.3
    # delivery: float = 1.0
    # biodist: float = 1.0
    # toxic: float = 0.05


def compute_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    mask: Dict[str, torch.Tensor],
    weights: Optional[LossWeights] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    计算多任务损失。
    
    Args:
        outputs: 模型输出
        targets: 真实标签
        mask: 有效样本掩码
        weights: 各任务权重
        
    Returns:
        (total_loss, loss_dict) 总损失和各任务损失
    """
    weights = weights or LossWeights()
    losses = {}
    total_loss = torch.tensor(0.0, device=next(iter(outputs.values())).device)
    
    # size: MSE loss
    if "size" in targets and mask["size"].any():
        m = mask["size"]
        pred = outputs["size"][m].squeeze(-1)
        tgt = targets["size"][m]
        losses["size"] = F.mse_loss(pred, tgt)
        total_loss = total_loss + weights.size * losses["size"]
    
    # delivery: MSE loss
    if "delivery" in targets and mask["delivery"].any():
        m = mask["delivery"]
        pred = outputs["delivery"][m].squeeze(-1)
        tgt = targets["delivery"][m]
        losses["delivery"] = F.mse_loss(pred, tgt)
        total_loss = total_loss + weights.delivery * losses["delivery"]
    
    # pdi: CrossEntropy
    if "pdi" in targets and mask["pdi"].any():
        m = mask["pdi"]
        pred = outputs["pdi"][m]
        tgt = targets["pdi"][m]
        losses["pdi"] = F.cross_entropy(pred, tgt)
        total_loss = total_loss + weights.pdi * losses["pdi"]
    
    # ee: CrossEntropy
    if "ee" in targets and mask["ee"].any():
        m = mask["ee"]
        pred = outputs["ee"][m]
        tgt = targets["ee"][m]
        losses["ee"] = F.cross_entropy(pred, tgt)
        total_loss = total_loss + weights.ee * losses["ee"]
    
    # toxic: CrossEntropy
    if "toxic" in targets and mask["toxic"].any():
        m = mask["toxic"]
        pred = outputs["toxic"][m]
        tgt = targets["toxic"][m]
        losses["toxic"] = F.cross_entropy(pred, tgt)
        total_loss = total_loss + weights.toxic * losses["toxic"]
    
    # biodist: KL divergence
    if "biodist" in targets and mask["biodist"].any():
        m = mask["biodist"]
        pred = outputs["biodist"][m]
        tgt = targets["biodist"][m]
        # KL divergence: KL(target || pred)
        losses["biodist"] = F.kl_div(
            pred.log().clamp(min=-100),
            tgt,
            reduction="batchmean",
        )
        total_loss = total_loss + weights.biodist * losses["biodist"]
    
    return total_loss, losses


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weights: Optional[LossWeights] = None,
) -> Dict[str, float]:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    task_losses = {k: 0.0 for k in ["size", "pdi", "ee", "delivery", "biodist", "toxic"]}
    n_batches = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        smiles = batch["smiles"]
        tabular = {k: v.to(device) for k, v in batch["tabular"].items()}
        targets = {k: v.to(device) for k, v in batch["targets"].items()}
        mask = {k: v.to(device) for k, v in batch["mask"].items()}
        
        optimizer.zero_grad()
        outputs = model(smiles, tabular)
        loss, losses = compute_multitask_loss(outputs, targets, mask, weights)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for k, v in losses.items():
            task_losses[k] += v.item()
        n_batches += 1
    
    return {
        "loss": total_loss / n_batches,
        **{f"loss_{k}": v / n_batches for k, v in task_losses.items() if v > 0},
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: Optional[LossWeights] = None,
) -> Dict[str, float]:
    """验证"""
    model.eval()
    total_loss = 0.0
    task_losses = {k: 0.0 for k in ["size", "pdi", "ee", "delivery", "biodist", "toxic"]}
    n_batches = 0
    
    # 用于计算准确率
    correct = {k: 0 for k in ["pdi", "ee", "toxic"]}
    total = {k: 0 for k in ["pdi", "ee", "toxic"]}
    
    for batch in tqdm(loader, desc="Validating", leave=False):
        smiles = batch["smiles"]
        tabular = {k: v.to(device) for k, v in batch["tabular"].items()}
        targets = {k: v.to(device) for k, v in batch["targets"].items()}
        mask = {k: v.to(device) for k, v in batch["mask"].items()}
        
        outputs = model(smiles, tabular)
        loss, losses = compute_multitask_loss(outputs, targets, mask, weights)
        
        total_loss += loss.item()
        for k, v in losses.items():
            task_losses[k] += v.item()
        n_batches += 1
        
        # 计算分类准确率
        for k in ["pdi", "ee", "toxic"]:
            if k in targets and mask[k].any():
                m = mask[k]
                pred = outputs[k][m].argmax(dim=-1)
                tgt = targets[k][m]
                correct[k] += (pred == tgt).sum().item()
                total[k] += m.sum().item()
    
    metrics = {
        "loss": total_loss / n_batches,
        **{f"loss_{k}": v / n_batches for k, v in task_losses.items() if v > 0},
    }
    
    # 添加准确率
    for k in ["pdi", "ee", "toxic"]:
        if total[k] > 0:
            metrics[f"acc_{k}"] = correct[k] / total[k]
    
    return metrics


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

