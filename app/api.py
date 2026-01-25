"""
FastAPI 配方优化 API

启动服务:
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from lnp_ml.config import MODELS_DIR
from lnp_ml.modeling.predict import load_model
from app.optimize import (
    optimize,
    format_results,
    AVAILABLE_ORGANS,
    TARGET_BIODIST,
)


# ============ Pydantic Models ============

class OptimizeRequest(BaseModel):
    """优化请求"""
    smiles: str = Field(..., description="Cationic lipid SMILES string")
    organ: str = Field(..., description="Target organ for optimization")
    top_k: int = Field(default=20, ge=1, le=100, description="Number of top formulations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "smiles": "CC(C)NCCNC(C)C",
                "organ": "liver",
                "top_k": 20
            }
        }


class FormulationResult(BaseModel):
    """单个配方结果"""
    rank: int
    target_biodist: float
    cationic_lipid_to_mrna_ratio: float
    cationic_lipid_mol_ratio: float
    phospholipid_mol_ratio: float
    cholesterol_mol_ratio: float
    peg_lipid_mol_ratio: float
    helper_lipid: str
    route: str
    all_biodist: Dict[str, float]


class OptimizeResponse(BaseModel):
    """优化响应"""
    smiles: str
    target_organ: str
    formulations: List[FormulationResult]
    message: str


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    device: str
    available_organs: List[str]


# ============ Global State ============

class ModelState:
    """模型状态管理"""
    model = None
    device = None
    model_path = None


state = ModelState()


# ============ Lifespan ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时加载模型"""
    # Startup
    logger.info("Starting API server...")
    
    # 确定设备
    if torch.cuda.is_available():
        device_str = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"
    
    # 可通过环境变量覆盖
    device_str = os.environ.get("DEVICE", device_str)
    state.device = torch.device(device_str)
    logger.info(f"Using device: {state.device}")
    
    # 加载模型
    model_path = Path(os.environ.get("MODEL_PATH", MODELS_DIR / "final" / "model.pt"))
    state.model_path = model_path
    
    logger.info(f"Loading model from {model_path}...")
    try:
        state.model = load_model(model_path, state.device)
        logger.success("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")
    state.model = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============ FastAPI App ============

app = FastAPI(
    title="LNP 配方优化 API",
    description="基于深度学习的 LNP 纳米颗粒配方优化服务",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Endpoints ============

@app.get("/", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy" if state.model is not None else "model_not_loaded",
        model_loaded=state.model is not None,
        device=str(state.device),
        available_organs=AVAILABLE_ORGANS,
    )


@app.get("/organs", response_model=List[str])
async def get_available_organs():
    """获取可用的目标器官列表"""
    return AVAILABLE_ORGANS


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_formulation(request: OptimizeRequest):
    """
    执行配方优化
    
    通过迭代式 Grid Search 寻找最大化目标器官 Biodistribution 的最优配方。
    """
    # 验证模型状态
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 验证器官
    if request.organ not in AVAILABLE_ORGANS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid organ: {request.organ}. Available: {AVAILABLE_ORGANS}"
        )
    
    # 验证 SMILES
    if not request.smiles or len(request.smiles.strip()) == 0:
        raise HTTPException(status_code=400, detail="SMILES string cannot be empty")
    
    logger.info(f"Optimization request: organ={request.organ}, smiles={request.smiles[:50]}...")
    
    try:
        # 执行优化
        results = optimize(
            smiles=request.smiles,
            organ=request.organ,
            model=state.model,
            device=state.device,
            top_k=request.top_k,
            batch_size=256,
        )
        
        # 转换结果
        formulations = []
        for i, f in enumerate(results):
            formulations.append(FormulationResult(
                rank=i + 1,
                target_biodist=f.get_biodist(request.organ),
                cationic_lipid_to_mrna_ratio=f.cationic_lipid_to_mrna_ratio,
                cationic_lipid_mol_ratio=f.cationic_lipid_mol_ratio,
                phospholipid_mol_ratio=f.phospholipid_mol_ratio,
                cholesterol_mol_ratio=f.cholesterol_mol_ratio,
                peg_lipid_mol_ratio=f.peg_lipid_mol_ratio,
                helper_lipid=f.helper_lipid,
                route=f.route,
                all_biodist={
                    col.replace("Biodistribution_", ""): f.biodist_predictions.get(col, 0.0)
                    for col in TARGET_BIODIST
                },
            ))
        
        logger.success(f"Optimization completed: {len(formulations)} formulations")
        
        return OptimizeResponse(
            smiles=request.smiles,
            target_organ=request.organ,
            formulations=formulations,
            message=f"Successfully found top {len(formulations)} formulations for {request.organ}",
        )
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

