# lnp-ml

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

LNP（脂质纳米颗粒）药物递送性能预测模型。

## 快速开始

### 1. 安装环境

```bash
pixi install
pixi shell
```

### 2. 数据处理

```bash
# 清洗原始数据 (raw -> interim)
make clean_data

# 处理内部数据集 (interim -> processed)
make data

# 处理外部预训练数据 (external -> processed)
make data_pretrain
```

### 3. 训练模型

**方式 A：直接训练（从零开始）**

```bash
make train
```

**方式 B：预训练 + 微调（推荐）**

利用外部 LiON 数据集（约 9000 条）进行预训练，再在内部数据上微调：

```bash
# Step 1: 处理外部数据
make data_pretrain

# Step 2: 在外部数据上预训练 backbone + delivery head
make pretrain

# Step 3: 加载预训练权重，在内部数据上多任务微调
make finetune
```

**方式 C：超参数调优**

```bash
make tune
```

### 4. 测试与预测

```bash
# 在测试集上评估
make test

# 生成预测结果
make predict
```

## 训练流程详解

### 预训练 (Pretrain)

在外部 LiON 数据上，仅训练 `quantified_delivery` 任务：

```bash
# 1. 先处理外部数据
python scripts/process_external.py

# 2. 预训练
python -m lnp_ml.modeling.pretrain \
    --train-path data/processed/train_pretrain.parquet \
    --val-path data/processed/val_pretrain.parquet \
    --epochs 50 \
    --lr 1e-4
```

产出:
- `data/processed/train_pretrain.parquet`: 处理后的训练数据
- `data/processed/val_pretrain.parquet`: 处理后的验证数据
- `models/pretrain_delivery.pt`: backbone + delivery head 权重
- `models/pretrain_history.json`: 训练历史

### 微调 (Finetune)

加载预训练权重，在内部多任务数据上训练：

```bash
python -m lnp_ml.modeling.train \
    --init-from-pretrain models/pretrain_delivery.pt \
    --load-delivery-head  # 可选：是否加载 delivery head 权重
```

产出:
- `models/model.pt`: 完整模型权重
- `models/history.json`: 训练历史

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         lnp_ml and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── lnp_ml   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes lnp_ml a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

