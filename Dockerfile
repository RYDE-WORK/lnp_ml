# LNP-ML Docker Image
# 多阶段构建，支持 API 和 Streamlit 两种服务

FROM python:3.8-slim AS base

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 复制项目代码
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
COPY lnp_ml/ ./lnp_ml/
COPY app/ ./app/

# 安装项目包
RUN pip install -e .

# 复制模型文件
COPY models/final/ ./models/final/

# ============ API 服务 ============
FROM base AS api

EXPOSE 8000

ENV MODEL_PATH=/app/models/final/model.pt

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ============ Streamlit 服务 ============
FROM base AS streamlit

EXPOSE 8501

# Streamlit 配置
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "app/app.py"]

