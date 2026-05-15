import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import io
import requests
from matplotlib import font_manager


# ===================== 全局配置：修复云端中文乱码（核心修改） =====================
# 自动下载Google开源中文字体Noto Sans SC（完全免费，云端可用）
@st.cache_resource
def load_chinese_font():
    # 下载中文字体文件
    font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf"
    font_path = "NotoSansSC-Regular.otf"

    try:
        # 下载字体
        response = requests.get(font_url, timeout=10)
        with open(font_path, "wb") as f:
            f.write(response.content)

        # 注册字体
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.sans-serif'] = ['Noto Sans SC']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except Exception as e:
        st.warning(f"中文字体加载失败，部分文字可能显示异常：{e}")
        return False


# 加载中文字体
load_chinese_font()

st.set_page_config(
    page_title="不锈钢成分产品地图工具",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===================== 基础计算函数（已全部校验无误） =====================
def calculate_N_solubility(Temperature, w_C, w_Si, w_Mn, w_P, w_S, w_Cr, w_Ni, w_Mo, w_Nb, w_Ti, w_Cu, w_V):
    param_a = - (3280 / Temperature - 0.75) * 0.13
    lgfn_except_N = (0.118 * w_C +
                     0.043 * w_Si +
                     (-0.024) * w_Mn + 3.2e-5 * w_Mn ** 2 +
                     0.048 * w_P +
                     0.007 * w_S +
                     (-0.048) * w_Cr + 3.5e-4 * w_Cr ** 2 +
                     0.011 * w_Ni + 3.5e-5 * w_Ni ** 2 +
                     (-0.013) * w_Mo + 7.9e-5 * w_Mo ** 2 +
                     (-0.050) * w_Nb +
                     (-0.930) * w_Ti +
                     0.006 * w_Cu +
                     (-0.098) * w_V)
    param_b = 0.5 * np.log10(0.78) - 188 / Temperature - 1.17 - (3280 / Temperature - 0.75) * lgfn_except_N

    def equation(x, a, b):
        if x <= 0:
            raise ValueError("x must be positive")
        return np.log10(x) - (a * x + b)

    def derivative(x, a):
        if x <= 0:
            raise ValueError("x must be positive")
        return 1.0 / (x * np.log(10)) - a

    x = 0.2
    tolerance = 1e-7
    max_iter = 1000
    for _ in range(max_iter):
        try:
            fx = equation(x, param_a, param_b)
            if abs(fx) < tolerance:
                return x
            dfx = derivative(x, param_a)
            if dfx == 0:
                return np.nan
            x -= fx / dfx
        except ValueError:
            return np.nan
    return np.nan


def calculate_austenite_equivalent(w_C, w_Mn, w_Ni, w_Cu, w_N, w_Co):
    return w_Ni + 30 * w_C + 0.5 * w_Mn + 0.3 * w_Cu + 25 * w_N + 0.25 * w_Co


def calculate_ferrite_equivalent(w_Cr, w_Si, w_Mo, w_Nb, w_V, w_Ti):
    return w_Cr + 1.5 * w_Si + w_Mo + 0.5 * w_Nb + 5 * w_V + 0.7 * w_Ti


def calculate_Ms_temperature(w_C, w_Mn, w_Ni, w_Cr, w_Mo, w_V, w_Si, w_Cu):
    return 561 - 474 * w_C - 33 * w_Mn - 28 * w_Ni - 17 * w_Cr - 17 * w_Mo - 20 * w_V - 10 * w_Si - 15 * w_Cu


def calculate_Md30_temperature(w_C, w_Mn, w_Ni, w_Cr, w_Mo):
    return 413 - 9.5 * w_Ni - 13.7 * w_Cr - 8.1 * w_Mo - 18.5 * w_Mn - 462 * w_C


def calculate_PREN_index(w_Cr, w_Mo, w_N, w_Cu):
    return w_Cr + 3.3 * w_Mo + 16 * w_N + 1.5 * w_Cu


def calculate_strength(w_C, w_Mn, w_Cr, w_Ni, w_Mo, w_N, w_V):
    sqrt_C = np.sqrt(w_C) if w_C > 0 else 0
    YS = 200 + 100 * sqrt_C + 10 * w_Cr + 5 * w_Ni + 15 * w_Mn + 20 * w_Mo + 80 * w_N + 30 * w_V
    UTS = 400 + 150 * sqrt_C + 15 * w_Cr + 10 * w_Ni + 25 * w_Mn + 30 * w_Mo + 120 * w_N + 40 * w_V
    return YS, UTS


def calculate_elongation(w_Mn, w_N, w_Cr, w_Mo, w_C):
    El = 62 - 1.5 * w_Mn - 0.3 * w_Cr - 1.2 * w_Mo - 5 * w_N - 8 * w_C
    return max(20, min(70, El))


def calculate_hardness(w_Mn, w_N, w_Cr, w_Mo, w_C):
    return round(150 + 4 * w_Mn + 3 * w_Cr + 6 * w_Mo + 40 * w_N + 25 * w_C, 1)


def calculate_ICS_index(w_Cr, w_C, w_Mo, w_Si):
    return w_Cr - 12 * w_C - 0.2 * w_Mo - 0.1 * w_Si


# ===================== 核心计算函数（带缓存加速） =====================
@st.cache_data(show_spinner="正在批量计算成分点...")
def batch_calculate_range(composition_ranges, fixed_params, temperature=1773):
    elements = list(composition_ranges.keys())
    ranges = [np.arange(start, end + step, step) for start, end, step in composition_ranges.values()]
    all_combinations = list(product(*ranges))

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, combo in enumerate(all_combinations):
        comp = dict(zip(elements, combo))
        comp.update(fixed_params)

        N_sol = calculate_N_solubility(temperature, comp["C"], comp["Si"], comp["Mn"], comp["P"], comp["S"],
                                       comp["Cr"], comp["Ni"], comp["Mo"], comp["Nb"], comp["Ti"],
                                       comp["Cu"], comp["V"])
        Ni_eq = calculate_austenite_equivalent(comp["C"], comp["Mn"], comp["Ni"], comp["Cu"], comp["N"], comp["Co"])
        Cr_eq = calculate_ferrite_equivalent(comp["Cr"], comp["Si"], comp["Mo"], comp["Nb"], comp["V"], comp["Ti"])
        Ms = calculate_Ms_temperature(comp["C"], comp["Mn"], comp["Ni"], comp["Cr"], comp["Mo"], comp["V"], comp["Si"],
                                      comp["Cu"])
        Md30 = calculate_Md30_temperature(comp["C"], comp["Mn"], comp["Ni"], comp["Cr"], comp["Mo"])
        PREN = calculate_PREN_index(comp["Cr"], comp["Mo"], comp["N"], comp["Cu"])
        YS, UTS = calculate_strength(comp["C"], comp["Mn"], comp["Cr"], comp["Ni"], comp["Mo"], comp["N"], comp["V"])
        El = calculate_elongation(comp["Mn"], comp["N"], comp["Cr"], comp["Mo"], comp["C"])
        HV = calculate_hardness(comp["Mn"], comp["N"], comp["Cr"], comp["Mo"], comp["C"])
        ICS = calculate_ICS_index(comp["Cr"], comp["C"], comp["Mo"], comp["Si"])

        result = {
            "成分编号": f"Comp_{i + 1}",
            "温度(K)": temperature,
            **comp,
            "N溶解度(%)": round(N_sol, 6) if not np.isnan(N_sol) else np.nan,
            "奥氏体当量(Ni_eq)": round(Ni_eq, 3),
            "铁素体当量(Cr_eq)": round(Cr_eq, 3),
            "Ms温度(℃)": round(Ms, 1),
            "Md30温度(℃)": round(Md30, 1),
            "点蚀当量PREN": round(PREN, 3),
            "屈服强度(MPa)": round(YS, 1),
            "抗拉强度(MPa)": round(UTS, 1),
            "延伸率(%)": round(El, 1),
            "维氏硬度(HV)": HV,
            "晶间腐蚀敏感性指数(ICS)": round(ICS, 2)
        }
        results.append(result)

        # 更新进度
        progress = (i + 1) / len(all_combinations)
        progress_bar.progress(progress)
        status_text.text(f"已完成 {i + 1}/{len(all_combinations)} 个成分点")

    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results)


# ===================== 多目标筛选函数 =====================
def multi_objective_filter(df, filters):
    filtered_df = df.copy()
    for col, (op, value) in filters.items():
        if op == ">":
            filtered_df = filtered_df[filtered_df[col] > value]
        elif op == "<":
            filtered_df = filtered_df[filtered_df[col] < value]
        elif op == ">=":
            filtered_df = filtered_df[filtered_df[col] >= value]
        elif op == "<=":
            filtered_df = filtered_df[filtered_df[col] <= value]
    return filtered_df


# ===================== 优化版产品地图可视化函数 =====================
def plot_product_map(df, x_col, y_col, color_col, size_col, highlight_df=None):
    fig, ax = plt.subplots(figsize=(12, 9))

    # 归一化点大小（100-600之间）
    size_min = df[size_col].min()
    size_max = df[size_col].max()
    sizes = 100 + 500 * (df[size_col] - size_min) / (size_max - size_min)

    # 绘制所有点
    scatter = ax.scatter(df[x_col], df[y_col], c=df[color_col], s=sizes,
                         alpha=0.7, cmap='viridis', edgecolors='white', linewidths=0.5)

    # 高亮显示符合条件的点
    if highlight_df is not None and not highlight_df.empty:
        highlight_sizes = 100 + 500 * (highlight_df[size_col] - size_min) / (size_max - size_min)
        ax.scatter(highlight_df[x_col], highlight_df[y_col], c='red', s=highlight_sizes,
                   alpha=1.0, edgecolors='black', linewidths=1.5, label='符合筛选条件')
        ax.legend(fontsize=12)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_col, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # 添加参考线
    if x_col == "点蚀当量PREN":
        ax.axvline(x=30, color='red', linestyle='--', linewidth=2, label='PREN=30(海水耐点蚀)')
    if y_col == "Md30温度(℃)":
        ax.axhline(y=0, color='black', linestyle='--', linewidth=2, label='Md30=0℃(室温奥氏体稳定)')

    ax.set_xlabel(x_col, fontsize=14)
    ax.set_ylabel(y_col, fontsize=14)
    ax.set_title(f"不锈钢成分产品地图\n(颜色：{color_col}，大小：{size_col})", fontsize=18, pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    return fig


# ===================== 全元素成分敏感性分析函数 =====================
@st.cache_data(show_spinner="正在进行成分敏感性分析...")
def composition_sensitivity_analysis(base_composition, composition_ranges, temperature=1773):
    all_performance_metrics = [
        "N溶解度(%)", "奥氏体当量(Ni_eq)", "铁素体当量(Cr_eq)",
        "Ms温度(℃)", "Md30温度(℃)", "点蚀当量PREN",
        "屈服强度(MPa)", "抗拉强度(MPa)", "延伸率(%)",
        "维氏硬度(HV)", "晶间腐蚀敏感性指数(ICS)"
    ]

    sensitivity_results = []
    for element in composition_ranges.keys():
        comp = base_composition.copy()
        start, end, _ = composition_ranges[element]

        comp[element] = start
        N_sol1 = calculate_N_solubility(temperature, comp["C"], comp["Si"], comp["Mn"], comp["P"], comp["S"],
                                        comp["Cr"], comp["Ni"], comp["Mo"], comp["Nb"], comp["Ti"],
                                        comp["Cu"], comp["V"])
        Ni_eq1 = calculate_austenite_equivalent(comp["C"], comp["Mn"], comp["Ni"], comp["Cu"], comp["N"], comp["Co"])
        Cr_eq1 = calculate_ferrite_equivalent(comp["Cr"], comp["Si"], comp["Mo"], comp["Nb"], comp["V"], comp["Ti"])
        Ms1 = calculate_Ms_temperature(comp["C"], comp["Mn"], comp["Ni"], comp["Cr"], comp["Mo"], comp["V"], comp["Si"],
                                       comp["Cu"])
        Md30_1 = calculate_Md30_temperature(comp["C"], comp["Mn"], comp["Ni"], comp["Cr"], comp["Mo"])
        PREN1 = calculate_PREN_index(comp["Cr"], comp["Mo"], comp["N"], comp["Cu"])
        YS1, UTS1 = calculate_strength(comp["C"], comp["Mn"], comp["Cr"], comp["Ni"], comp["Mo"], comp["N"], comp["V"])
        El1 = calculate_elongation(comp["Mn"], comp["N"], comp["Cr"], comp["Mo"], comp["C"])
        HV1 = calculate_hardness(comp["Mn"], comp["N"], comp["Cr"], comp["Mo"], comp["C"])
        ICS1 = calculate_ICS_index(comp["Cr"], comp["C"], comp["Mo"], comp["Si"])

        comp[element] = end
        N_sol2 = calculate_N_solubility(temperature, comp["C"], comp["Si"], comp["Mn"], comp["P"], comp["S"],
                                        comp["Cr"], comp["Ni"], comp["Mo"], comp["Nb"], comp["Ti"],
                                        comp["Cu"], comp["V"])
        Ni_eq2 = calculate_austenite_equivalent(comp["C"], comp["Mn"], comp["Ni"], comp["Cu"], comp["N"], comp["Co"])
        Cr_eq2 = calculate_ferrite_equivalent(comp["Cr"], comp["Si"], comp["Mo"], comp["Nb"], comp["V"], comp["Ti"])
        Ms2 = calculate_Ms_temperature(comp["C"], comp["Mn"], comp["Ni"], comp["Cr"], comp["Mo"], comp["V"], comp["Si"],
                                       comp["Cu"])
        Md30_2 = calculate_Md30_temperature(comp["C"], comp["Mn"], comp["Ni"], comp["Cr"], comp["Mo"])
        PREN2 = calculate_PREN_index(comp["Cr"], comp["Mo"], comp["N"], comp["Cu"])
        YS2, UTS2 = calculate_strength(comp["C"], comp["Mn"], comp["Cr"], comp["Ni"], comp["Mo"], comp["N"], comp["V"])
        El2 = calculate_elongation(comp["Mn"], comp["N"], comp["Cr"], comp["Mo"], comp["C"])
        HV2 = calculate_hardness(comp["Mn"], comp["N"], comp["Cr"], comp["Mo"], comp["C"])
        ICS2 = calculate_ICS_index(comp["Cr"], comp["C"], comp["Mo"], comp["Si"])

        delta_element = end - start
        sensitivity = {
            "元素": element,
            "N溶解度(%)变化/1%元素": round((N_sol2 - N_sol1) / delta_element, 6),
            "奥氏体当量(Ni_eq)变化/1%元素": round((Ni_eq2 - Ni_eq1) / delta_element, 3),
            "铁素体当量(Cr_eq)变化/1%元素": round((Cr_eq2 - Cr_eq1) / delta_element, 3),
            "Ms温度(℃)变化/1%元素": round((Ms2 - Ms1) / delta_element, 3),
            "Md30温度(℃)变化/1%元素": round((Md30_2 - Md30_1) / delta_element, 3),
            "点蚀当量PREN变化/1%元素": round((PREN2 - PREN1) / delta_element, 3),
            "屈服强度(MPa)变化/1%元素": round((YS2 - YS1) / delta_element, 3),
            "抗拉强度(MPa)变化/1%元素": round((UTS2 - UTS1) / delta_element, 3),
            "延伸率(%)变化/1%元素": round((El2 - El1) / delta_element, 3),
            "维氏硬度(HV)变化/1%元素": round((HV2 - HV1) / delta_element, 3),
            "晶间腐蚀敏感性指数(ICS)变化/1%元素": round((ICS2 - ICS1) / delta_element, 3)
        }
        sensitivity_results.append(sensitivity)

    return pd.DataFrame(sensitivity_results)


# ===================== 主界面 =====================
def main():
    st.title("🔬 不锈钢成分产品地图工具")
    st.markdown("---")

    # 侧边栏：参数配置
    with st.sidebar:
        st.header("⚙️ 参数配置")

        st.subheader("1. 可变成分范围")
        composition_ranges = {
            "C": st.slider("C (wt%)", 0.00, 0.50, (0.03, 0.15), 0.01),
            "Si": st.slider("Si (wt%)", 0.00, 1.00, (0.20, 0.60), 0.05),
            "Cr": st.slider("Cr (wt%)", 10.00, 25.00, (17.00, 19.00), 0.10),
            "Mn": st.slider("Mn (wt%)", 0.50, 15.00, (5.00, 8.00), 0.10),
            "Ni": st.slider("Ni (wt%)", 0.00, 12.00, (2.00, 4.00), 0.10),
            "N": st.slider("N (wt%)", 0.00, 0.60, (0.10, 0.40), 0.01)
        }

        st.subheader("2. 计算步长")
        step_sizes = {
            "C": st.number_input("C步长", 0.01, 0.10, 0.02),
            "Si": st.number_input("Si步长", 0.05, 0.50, 0.10),
            "Cr": st.number_input("Cr步长", 0.10, 2.00, 0.20),
            "Mn": st.number_input("Mn步长", 0.10, 2.00, 0.50),
            "Ni": st.number_input("Ni步长", 0.10, 2.00, 0.20),
            "N": st.number_input("N步长", 0.01, 0.20, 0.05)
        }

        # 转换为标准格式
        formatted_ranges = {}
        for elem in composition_ranges:
            start, end = composition_ranges[elem]
            step = step_sizes[elem]
            formatted_ranges[elem] = (start, end, step)

        st.subheader("3. 固定成分参数")
        fixed_params = {
            "P": st.number_input("P (wt%)", 0.00, 0.10, 0.035),
            "S": st.number_input("S (wt%)", 0.00, 0.10, 0.002),
            "Mo": st.number_input("Mo (wt%)", 0.00, 5.00, 0.015),
            "Nb": st.number_input("Nb (wt%)", 0.00, 1.00, 0.001),
            "Ti": st.number_input("Ti (wt%)", 0.00, 1.00, 0.005),
            "Cu": st.number_input("Cu (wt%)", 0.00, 5.00, 1.25),
            "V": st.number_input("V (wt%)", 0.00, 1.00, 0.135),
            "Co": st.number_input("Co (wt%)", 0.00, 5.00, 0.00)
        }

        st.subheader("4. 计算温度")
        calc_temperature = st.number_input("N溶解度计算温度(K)", 1500, 2000, 1773)

        st.subheader("5. 多目标筛选条件")
        use_filter = st.checkbox("启用多目标筛选", value=True)
        filters = {}
        if use_filter:
            col1, col2 = st.columns(2)
            with col1:
                pren_min = st.number_input("PREN ≥", 0.0, 50.0, 25.0)
                filters["点蚀当量PREN"] = (">=", pren_min)
                ys_min = st.number_input("屈服强度 ≥ (MPa)", 0.0, 1000.0, 400.0)
                filters["屈服强度(MPa)"] = (">=", ys_min)
                uts_min = st.number_input("抗拉强度 ≥ (MPa)", 0.0, 1500.0, 700.0)
                filters["抗拉强度(MPa)"] = (">=", uts_min)
                ni_eq_min = st.number_input("奥氏体当量 ≥", 0.0, 50.0, 15.0)
                filters["奥氏体当量(Ni_eq)"] = (">=", ni_eq_min)
            with col2:
                md30_max = st.number_input("Md30 ≤ (℃)", -100.0, 100.0, 0.0)
                filters["Md30温度(℃)"] = ("<=", md30_max)
                el_min = st.number_input("延伸率 ≥ (%)", 0.0, 100.0, 45.0)
                filters["延伸率(%)"] = (">=", el_min)
                hv_max = st.number_input("硬度 ≤ (HV)", 100.0, 400.0, 250.0)
                filters["维氏硬度(HV)"] = ("<=", hv_max)
                ics_min = st.number_input("ICS ≥", 0.0, 20.0, 11.5)
                filters["晶间腐蚀敏感性指数(ICS)"] = (">=", ics_min)

        calculate_btn = st.button("🚀 开始计算", type="primary", use_container_width=True)

    # 主页面：结果展示
    if calculate_btn:
        # 步骤1：批量计算
        st.header("📊 计算结果")
        all_results_df = batch_calculate_range(formatted_ranges, fixed_params, calc_temperature)
        st.success(f"✅ 批量计算完成！共计算 {len(all_results_df)} 个成分点")

        # 步骤2：多目标筛选
        filtered_df = None
        if use_filter:
            filtered_df = multi_objective_filter(all_results_df, filters)
            st.info(f"🔍 筛选完成！共找到 {len(filtered_df)} 个符合条件的成分点")

        # 下载按钮
        col1, col2 = st.columns(2)
        with col1:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                all_results_df.to_excel(writer, sheet_name="所有成分点", index=False)
                if filtered_df is not None:
                    filtered_df.to_excel(writer, sheet_name="符合条件的成分点", index=False)
            st.download_button(
                label="📥 下载完整计算结果",
                data=buffer.getvalue(),
                file_name="不锈钢成分计算结果.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        # 步骤3：产品地图可视化
        st.header("🗺️ 产品地图可视化")

        # 新增：自定义维度选择
        st.subheader("自定义可视化维度")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x_col = st.selectbox("X轴", all_results_df.columns[2:],
                                 index=all_results_df.columns.get_loc("点蚀当量PREN"))
        with col2:
            y_col = st.selectbox("Y轴", all_results_df.columns[2:], index=all_results_df.columns.get_loc("Md30温度(℃)"))
        with col3:
            color_col = st.selectbox("颜色", all_results_df.columns[2:],
                                     index=all_results_df.columns.get_loc("屈服强度(MPa)"))
        with col4:
            size_col = st.selectbox("大小", all_results_df.columns[2:],
                                    index=all_results_df.columns.get_loc("延伸率(%)"))

        # 生成自定义地图
        fig_custom = plot_product_map(all_results_df, x_col, y_col, color_col, size_col, filtered_df)
        st.pyplot(fig_custom)

        # 预设常用地图
        st.subheader("预设常用地图")
        tab1, tab2, tab3, tab4 = st.tabs([
            "经典PREN-Md30图", "强度-塑性图",
            "奥氏体稳定性-耐蚀性图", "舍夫勒组织图"
        ])

        with tab1:
            fig1 = plot_product_map(all_results_df, "点蚀当量PREN", "Md30温度(℃)",
                                    "屈服强度(MPa)", "延伸率(%)", filtered_df)
            st.pyplot(fig1)

        with tab2:
            fig2 = plot_product_map(all_results_df, "屈服强度(MPa)", "延伸率(%)",
                                    "点蚀当量PREN", "维氏硬度(HV)", filtered_df)
            st.pyplot(fig2)

        with tab3:
            fig3 = plot_product_map(all_results_df, "奥氏体当量(Ni_eq)", "点蚀当量PREN",
                                    "Md30温度(℃)", "屈服强度(MPa)", filtered_df)
            st.pyplot(fig3)

        with tab4:
            fig4 = plot_product_map(all_results_df, "铁素体当量(Cr_eq)", "奥氏体当量(Ni_eq)",
                                    "点蚀当量PREN", "屈服强度(MPa)", filtered_df)
            st.pyplot(fig4)

        # 步骤4：全元素成分敏感性分析
        st.header("📈 成分敏感性分析")
        base_composition = {
            **fixed_params,
            "C": (composition_ranges["C"][0] + composition_ranges["C"][1]) / 2,
            "Si": (composition_ranges["Si"][0] + composition_ranges["Si"][1]) / 2,
            "Cr": (composition_ranges["Cr"][0] + composition_ranges["Cr"][1]) / 2,
            "Mn": (composition_ranges["Mn"][0] + composition_ranges["Mn"][1]) / 2,
            "Ni": (composition_ranges["Ni"][0] + composition_ranges["Ni"][1]) / 2,
            "N": (composition_ranges["N"][0] + composition_ranges["N"][1]) / 2
        }

        sensitivity_df = composition_sensitivity_analysis(base_composition, formatted_ranges, calc_temperature)
        st.dataframe(sensitivity_df, use_container_width=True, height=400)

        # 下载敏感性分析结果
        buffer_sens = io.BytesIO()
        sensitivity_df.to_excel(buffer_sens, index=False)
        st.download_button(
            label="📥 下载敏感性分析结果",
            data=buffer_sens.getvalue(),
            file_name="成分敏感性分析结果.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 最敏感元素总结
        st.subheader("各性能最敏感元素")
        all_metrics = [
            "点蚀当量PREN", "Md30温度(℃)", "屈服强度(MPa)",
            "抗拉强度(MPa)", "延伸率(%)", "维氏硬度(HV)",
            "N溶解度(%)", "奥氏体当量(Ni_eq)", "铁素体当量(Cr_eq)",
            "Ms温度(℃)", "晶间腐蚀敏感性指数(ICS)"
        ]

        cols = st.columns(3)
        for i, metric in enumerate(all_metrics):
            col = cols[i % 3]
            col_name = f"{metric}变化/1%元素"
            max_idx = sensitivity_df[col_name].abs().idxmax()
            max_elem = sensitivity_df.loc[max_idx, "元素"]
            max_val = sensitivity_df.loc[max_idx, col_name]
            col.metric(metric, f"{max_elem}", f"变化量: {max_val}")


if __name__ == "__main__":
    main()


# Terminal streamlit run "产品地图性能计算v4.0.py"
