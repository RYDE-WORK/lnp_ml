"""
Streamlit 配方优化交互界面

启动应用:
    streamlit run app/app.py
"""

import io
from datetime import datetime

import httpx
import pandas as pd
import streamlit as st

# ============ 配置 ============

API_URL = "http://localhost:8000"

AVAILABLE_ORGANS = [
    "liver",
    "spleen", 
    "lung",
    "heart",
    "kidney",
    "muscle",
    "lymph_nodes",
]

ORGAN_LABELS = {
    "liver": "肝脏 (Liver)",
    "spleen": "脾脏 (Spleen)",
    "lung": "肺 (Lung)",
    "heart": "心脏 (Heart)",
    "kidney": "肾脏 (Kidney)",
    "muscle": "肌肉 (Muscle)",
    "lymph_nodes": "淋巴结 (Lymph Nodes)",
}

# ============ 页面配置 ============

st.set_page_config(
    page_title="LNP 配方优化",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ 自定义样式 ============

st.markdown("""
<style>
    /* 主标题样式 */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* 副标题样式 */
    .sub-title {
        font-size: 1.1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* 结果卡片 */
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* 指标高亮 */
    .metric-highlight {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* 侧边栏样式 */
    .sidebar-section {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* 状态指示器 */
    .status-online {
        color: #28a745;
        font-weight: 600;
    }
    
    .status-offline {
        color: #dc3545;
        font-weight: 600;
    }
    
    /* 表格样式优化 */
    .dataframe {
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ============ 辅助函数 ============

def check_api_status() -> bool:
    """检查 API 状态"""
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{API_URL}/")
            return response.status_code == 200
    except:
        return False


def call_optimize_api(smiles: str, organ: str, top_k: int = 20) -> dict:
    """调用优化 API"""
    with httpx.Client(timeout=300) as client:  # 5 分钟超时
        response = client.post(
            f"{API_URL}/optimize",
            json={
                "smiles": smiles,
                "organ": organ,
                "top_k": top_k,
            },
        )
        response.raise_for_status()
        return response.json()


def format_results_dataframe(results: dict) -> pd.DataFrame:
    """将 API 结果转换为 DataFrame"""
    formulations = results["formulations"]
    target_organ = results["target_organ"]
    
    rows = []
    for f in formulations:
        row = {
            "排名": f["rank"],
            # f"{target_organ}分布": f"{f['target_biodist']*100:.2f}%",
            f"{target_organ}分布": f"{f['target_biodist']*100:.8f}%",
            "阳离子脂质/mRNA比例": f["cationic_lipid_to_mrna_ratio"],
            "阳离子脂质(mol)比例": f["cationic_lipid_mol_ratio"],
            "磷脂(mol)比例": f["phospholipid_mol_ratio"],
            "胆固醇(mol)比例": f["cholesterol_mol_ratio"],
            "PEG脂质(mol)比例": f["peg_lipid_mol_ratio"],
            "辅助脂质": f["helper_lipid"],
            "给药途径": f["route"],
        }
        # 添加其他器官的 biodist
        for organ, value in f["all_biodist"].items():
            if organ != target_organ:
                row[f"{organ}分布"] = f"{value*100:.2f}%"
        rows.append(row)
    
    return pd.DataFrame(rows)


def create_export_csv(df: pd.DataFrame, smiles: str, organ: str) -> str:
    """创建导出用的 CSV 内容"""
    # 添加元信息
    meta_info = f"# LNP 配方优化结果\n# SMILES: {smiles}\n# 目标器官: {organ}\n# 导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    csv_content = df.to_csv(index=False)
    return meta_info + csv_content


# ============ 主界面 ============

def main():
    # 标题
    st.markdown('<h1 class="main-title">🧬 LNP 配方优化系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">基于深度学习的脂质纳米颗粒配方智能优选</p>', unsafe_allow_html=True)
    
    # 检查 API 状态
    api_online = check_api_status()
    
    # ========== 侧边栏 ==========
    with st.sidebar:
        # st.header("⚙️ 参数设置")
        
        # API 状态
        if api_online:
            st.success("🟢 API 服务在线")
        else:
            st.error("🔴 API 服务离线")
            st.info("请先启动 API 服务:\n```\nuvicorn app.api:app --port 8000\n```")
        
        # st.divider()
        
        # SMILES 输入
        st.subheader("🔬 分子结构")
        smiles_input = st.text_area(
            "输入阳离子脂质 SMILES",
            value="",
            height=100,
            placeholder="例如: CC(C)NCCNC(C)C",
            help="输入阳离子脂质的 SMILES 字符串",
        )
        
        # 示例 SMILES
        # with st.expander("📋 示例 SMILES"):
        #     example_smiles = {
        #         "DLin-MC3-DMA": "CC(C)=CCCC(C)=CCCC(C)=CCN(C)CCCCCCCCOC(=O)CCCCCCC/C=C\\CCCCCCCC",
        #         "简单胺": "CC(C)NCCNC(C)C",
        #         "长链胺": "CCCCCCCCCCCCNCCNCCCCCCCCCCCC",
        #     }
        #     for name, smi in example_smiles.items():
        #         if st.button(f"使用 {name}", key=f"example_{name}"):
        #             st.session_state["smiles_input"] = smi
        #             st.rerun()
        
        # st.divider()
        
        # 目标器官选择
        st.subheader("🎯 目标器官")
        selected_organ = st.selectbox(
            "选择优化目标器官",
            options=AVAILABLE_ORGANS,
            format_func=lambda x: ORGAN_LABELS.get(x, x),
            index=0,
        )
        
        # st.divider()
        
        # 高级选项
        with st.expander("🔧 高级选项"):
            top_k = st.slider(
                "返回配方数量",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
            )
        
        st.divider()
        
        # 优化按钮
        optimize_button = st.button(
            "🚀 开始配方优选",
            type="primary",
            use_container_width=True,
            disabled=not api_online or not smiles_input.strip(),
        )
    
    # ========== 主内容区 ==========
    
    # 使用 session state 存储结果
    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "results_df" not in st.session_state:
        st.session_state["results_df"] = None
    
    # 执行优化
    if optimize_button and smiles_input.strip():
        with st.spinner("🔄 正在优化配方，请稍候..."):
            try:
                results = call_optimize_api(
                    smiles=smiles_input.strip(),
                    organ=selected_organ,
                    top_k=top_k,
                )
                st.session_state["results"] = results
                st.session_state["results_df"] = format_results_dataframe(results)
                st.session_state["smiles_used"] = smiles_input.strip()
                st.session_state["organ_used"] = selected_organ
                st.success("✅ 优化完成！")
            except httpx.RequestError as e:
                st.error(f"❌ API 请求失败: {e}")
            except Exception as e:
                st.error(f"❌ 发生错误: {e}")
    
    # 显示结果
    if st.session_state["results"] is not None:
        results = st.session_state["results"]
        df = st.session_state["results_df"]
        
        # 结果概览
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "目标器官",
                ORGAN_LABELS.get(results["target_organ"], results["target_organ"]).split(" ")[0],
            )
        
        with col2:
            best_score = results["formulations"][0]["target_biodist"]
            st.metric(
                "最优分布",
                f"{best_score*100:.2f}%",
            )
        
        with col3:
            st.metric(
                "优选配方数",
                len(results["formulations"]),
            )
        
        st.divider()
        
        # 结果表格
        st.subheader("📊 优选配方列表")
        
        # 导出按钮行
        col_export, col_spacer = st.columns([1, 4])
        with col_export:
            csv_content = create_export_csv(
                df,
                st.session_state.get("smiles_used", ""),
                st.session_state.get("organ_used", ""),
            )
            st.download_button(
                label="📥 导出 CSV",
                data=csv_content,
                file_name=f"lnp_optimization_{results['target_organ']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        
        # 显示表格
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=600,
        )
        
        # 详细信息
        # with st.expander("🔍 查看最优配方详情"):
        #     best = results["formulations"][0]
            
        #     col1, col2 = st.columns(2)
            
        #     with col1:
        #         st.markdown("**配方参数**")
        #         st.json({
        #             "阳离子脂质/mRNA 比例": best["cationic_lipid_to_mrna_ratio"],
        #             "阳离子脂质 (mol%)": best["cationic_lipid_mol_ratio"],
        #             "磷脂 (mol%)": best["phospholipid_mol_ratio"],
        #             "胆固醇 (mol%)": best["cholesterol_mol_ratio"],
        #             "PEG 脂质 (mol%)": best["peg_lipid_mol_ratio"],
        #             "辅助脂质": best["helper_lipid"],
        #             "给药途径": best["route"],
        #         })
            
        #     with col2:
        #         st.markdown("**各器官 Biodistribution 预测**")
        #         biodist_df = pd.DataFrame([
        #             {"器官": ORGAN_LABELS.get(k, k), "Biodistribution": f"{v:.4f}"}
        #             for k, v in best["all_biodist"].items()
        #         ])
        #         st.dataframe(biodist_df, hide_index=True, use_container_width=True)
    
    else:
        # 欢迎信息
        st.info("👈 请在左侧输入 SMILES 并选择目标器官，然后点击「开始配方优选」")
        
        # 使用说明
        # with st.expander("📖 使用说明"):
        #     st.markdown("""
        #     ### 如何使用
            
        #     1. **输入 SMILES**: 在左侧输入框中输入阳离子脂质的 SMILES 字符串
        #     2. **选择目标器官**: 选择您希望优化的器官靶向
        #     3. **点击优选**: 系统将自动搜索最优配方组合
        #     4. **查看结果**: 右侧将显示 Top-20 优选配方
        #     5. **导出数据**: 点击导出按钮将结果保存为 CSV 文件
            
        #     ### 优化参数
            
        #     系统会优化以下配方参数:
        #     - **阳离子脂质/mRNA 比例**: 0.05 - 0.30
        #     - **阳离子脂质 mol 比例**: 0.05 - 0.80
        #     - **磷脂 mol 比例**: 0.00 - 0.80
        #     - **胆固醇 mol 比例**: 0.00 - 0.80
        #     - **PEG 脂质 mol 比例**: 0.00 - 0.05
        #     - **辅助脂质**: DOPE / DSPC / DOTAP
        #     - **给药途径**: 静脉注射 / 肌肉注射
            
        #     ### 约束条件
            
        #     mol 比例之和 = 1 (阳离子脂质 + 磷脂 + 胆固醇 + PEG 脂质)
        #     """)


if __name__ == "__main__":
    main()

