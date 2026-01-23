"""训练过程可视化工具：绘制 loss 曲线"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免 GUI 依赖

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_loss_curves(
    history: Union[Dict[str, List[Dict]], List[Dict]],
    output_path: Path,
    title: str = "Training Loss Curves",
    figsize: tuple = (12, 8),
) -> None:
    """
    绘制训练过程中各个 loss 组成部分的变化曲线。

    支持两种 history 格式：
    1. 预训练格式（单任务）：
       {"train": [{"loss": 0.1, ...}, ...], "val": [{"loss": 0.1, ...}, ...]}
    2. CV fold 格式：
       [{"epoch": 1, "train_loss": 0.1, "val_loss": 0.1, ...}, ...]

    Args:
        history: 训练历史记录
        output_path: 输出图片路径
        title: 图标题
        figsize: 图片尺寸
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 判断 history 格式并统一处理
    if isinstance(history, dict) and "train" in history:
        # 格式1: {"train": [...], "val": [...]}
        _plot_standard_history(history, output_path, title, figsize)
    elif isinstance(history, list):
        # 格式2: [{...}, {...}, ...]
        _plot_flat_history(history, output_path, title, figsize)
    else:
        raise ValueError(f"Unsupported history format: {type(history)}")


def _plot_standard_history(
    history: Dict[str, List[Dict]],
    output_path: Path,
    title: str,
    figsize: tuple,
) -> None:
    """绘制标准格式的 history（train/val 分开）"""
    train_history = history.get("train", [])
    val_history = history.get("val", [])

    if not train_history:
        return

    epochs = list(range(1, len(train_history) + 1))

    # 收集所有 loss 键
    all_loss_keys = set()
    for record in train_history + val_history:
        for key in record.keys():
            if key.startswith("loss") or key == "loss":
                all_loss_keys.add(key)

    # 分离总 loss 和各任务 loss
    total_loss_key = "loss"
    task_loss_keys = sorted([k for k in all_loss_keys if k != "loss"])

    # 创建子图
    n_subplots = 1 + (1 if task_loss_keys else 0)
    fig, axes = plt.subplots(n_subplots, 1, figsize=(figsize[0], figsize[1] * n_subplots / 2))
    if n_subplots == 1:
        axes = [axes]

    # 颜色配置
    colors = plt.cm.tab10.colors

    # 子图1：总 loss
    ax = axes[0]
    train_total_loss = [r.get(total_loss_key, 0) for r in train_history]
    val_total_loss = [r.get(total_loss_key, 0) for r in val_history]

    ax.plot(epochs, train_total_loss, 'o-', label='Train Total Loss', color=colors[0], linewidth=2, markersize=4)
    if val_total_loss:
        ax.plot(epochs, val_total_loss, 's--', label='Val Total Loss', color=colors[1], linewidth=2, markersize=4)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{title} - Total Loss', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, len(epochs) + 0.5)

    # 子图2：各任务 loss（如果有）
    if task_loss_keys:
        ax = axes[1]
        for i, key in enumerate(task_loss_keys):
            task_name = key.replace("loss_", "").upper()
            train_values = [r.get(key, 0) for r in train_history]
            val_values = [r.get(key, 0) for r in val_history]

            color = colors[i % len(colors)]
            ax.plot(epochs, train_values, 'o-', label=f'Train {task_name}', color=color, alpha=0.8, linewidth=1.5, markersize=3)
            if val_values and any(v > 0 for v in val_values):
                ax.plot(epochs, val_values, 's--', label=f'Val {task_name}', color=color, alpha=0.5, linewidth=1.5, markersize=3)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{title} - Per-Task Loss', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, len(epochs) + 0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _plot_flat_history(
    history: List[Dict],
    output_path: Path,
    title: str,
    figsize: tuple,
) -> None:
    """绘制扁平格式的 history（CV fold 格式）"""
    if not history:
        return

    epochs = [r.get("epoch", i + 1) for i, r in enumerate(history)]

    # 收集所有 loss 相关的键
    loss_keys = set()
    for record in history:
        for key in record.keys():
            if "loss" in key.lower():
                loss_keys.add(key)

    # 分类
    train_keys = sorted([k for k in loss_keys if "train" in k.lower()])
    val_keys = sorted([k for k in loss_keys if "val" in k.lower()])

    # 创建子图
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    colors = plt.cm.tab10.colors
    color_idx = 0

    # 绘制训练 loss
    for key in train_keys:
        values = [r.get(key, 0) for r in history]
        label = key.replace("_", " ").title()
        ax.plot(epochs, values, 'o-', label=label, color=colors[color_idx % len(colors)], 
                linewidth=2, markersize=4, alpha=0.9)
        color_idx += 1

    # 绘制验证 loss
    for key in val_keys:
        values = [r.get(key, 0) for r in history]
        label = key.replace("_", " ").title()
        ax.plot(epochs, values, 's--', label=label, color=colors[color_idx % len(colors)], 
                linewidth=2, markersize=4, alpha=0.7)
        color_idx += 1

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, len(epochs) + 0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_multitask_loss_curves(
    history: Dict[str, List[Dict]],
    output_path: Path,
    title: str = "Multi-task Training Loss",
    figsize: tuple = (14, 10),
) -> None:
    """
    专门用于多任务训练的 loss 曲线绘制。
    
    将各个任务的 loss 分别绘制在不同的子图中，便于比较。
    
    Args:
        history: {"train": [...], "val": [...]} 格式的训练历史
        output_path: 输出路径
        title: 图标题
        figsize: 图片尺寸
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_history = history.get("train", [])
    val_history = history.get("val", [])

    if not train_history:
        return

    epochs = list(range(1, len(train_history) + 1))

    # 提取所有任务的 loss 键
    task_keys = set()
    for record in train_history:
        for key in record.keys():
            if key.startswith("loss_"):
                task_name = key.replace("loss_", "")
                task_keys.add(task_name)

    task_keys = sorted(task_keys)

    # 计算子图布局
    n_tasks = len(task_keys)
    if n_tasks == 0:
        # 只有总 loss，使用简单绘图
        _plot_standard_history(history, output_path, title, figsize)
        return

    # 包含总 loss，共 n_tasks + 1 个子图
    n_plots = n_tasks + 1
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows / 2))
    if n_plots == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]

    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    colors = plt.cm.tab10.colors

    # 子图1：总 loss
    ax = axes_flat[0]
    train_total = [r.get("loss", 0) for r in train_history]
    val_total = [r.get("loss", 0) for r in val_history]

    ax.plot(epochs, train_total, 'o-', label='Train', color=colors[0], linewidth=2, markersize=4)
    if val_total:
        ax.plot(epochs, val_total, 's--', label='Val', color=colors[1], linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 各任务子图
    for idx, task in enumerate(task_keys):
        ax = axes_flat[idx + 1]
        key = f"loss_{task}"

        train_values = [r.get(key, 0) for r in train_history]
        val_values = [r.get(key, 0) for r in val_history]

        # 只绘制有值的数据
        if any(v > 0 for v in train_values):
            ax.plot(epochs, train_values, 'o-', label='Train', color=colors[0], linewidth=2, markersize=4)
        if val_values and any(v > 0 for v in val_values):
            ax.plot(epochs, val_values, 's--', label='Val', color=colors[1], linewidth=2, markersize=4)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{task.upper()} Loss', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

