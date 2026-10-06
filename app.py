import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import urllib.request
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit 페이지 설정
st.set_page_config(page_title="서울 기온 선형회귀 예측기", layout="wide")

# --- 한글 폰트 설정 (폰트 파일 자동 다운로드 및 적용) ---
@st.cache_resource
def setup_korean_font():
    font_path = "NanumGothic.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        try:
            urllib.request.urlretrieve(url, font_path)
        except Exception:
            pass

    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rc('font', family='NanumGothic')
    else:
        font_names = [f.name for f in fm.fontManager.ttflist]
        if 'NanumGothic' in font_names:
            plt.rc('font', family='NanumGothic')
        elif 'Malgun Gothic' in font_names:
            plt.rc('font', family='Malgun Gothic')
        elif 'AppleGothic' in font_names:
            plt.rc('font', family='AppleGothic')

    plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()

# 커스텀 CSS (안내 박스 및 메트릭 레이아웃)
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
    }
    .info-box-blue {
        background-color: #eff6ff;
        border-radius: 8px;
        padding: 16px;
        color: #1d4ed8;
        font-size: 14.5px;
        line-height: 1.6;
        margin-top: 15px;
        margin-bottom: 25px;
    }
    .warning-box {
        background-color: #fefde8;
        border-radius: 8px;
        padding: 16px;
        color: #854d0e;
        font-size: 14.5px;
        line-height: 1.6;
        margin-top: 10px;
        margin-bottom: 25px;
    }
    .metric-subtext {
        font-size: 12px;
        color: #888888;
        margin-top: -10px;
    }
</style>
""", unsafe_allow_html=True)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_process_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = df.columns.str.strip()
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    # 2025년 이하 데이터만 필터링
    df = df[df['연도'] <= 2025]
    
    yearly_stats = df.groupby('연도').agg(
        관측일수=('평균기온', 'count'),
        연평균기온=('평균기온', 'mean')
    ).reset_index()
    
    valid_data = yearly_stats[yearly_stats['관측일수'] >= 300].copy()
    return valid_data

try:
    data = load_and_process_data()
    
    # 메인 타이틀 & 설명
    st.title("🌡️ 서울 기온 선형회귀 예측기")
    st.caption("서울의 연평균기온으로 선형회귀 모델을 만들고, 과거 자료로 학습한 모델이 최근 기온을 얼마나 잘 예측하는지 평가합니다.")
    st.markdown("---")

    # ==========================================
    # ① 훈련 데이터와 테스트 데이터
    # ==========================================
    st.markdown("### ① 훈련 데이터와 테스트 데이터")

    # 데이터 분할 및 개수 계산
    test_data = data[(data['연도'] >= 2006) & (data['연도'] <= 2025)].copy()
    train_50y = data[(data['연도'] >= 1956) & (data['연도'] <= 2005)].copy()
    train_100y = data[(data['연도'] >= 1906) & (data['연도'] <= 2005)].copy()

    cnt_test = len(test_data)
    cnt_50y = len(train_50y)
    cnt_100y = len(train_100y)

    c_t1, c_t2, c_t3 = st.columns(3)
    
    with c_t1:
        st.metric("테스트 데이터", f"{cnt_test}개년")
        st.markdown("<div class='metric-subtext'>2006~2025년</div>", unsafe_allow_html=True)
        
    with c_t2:
        st.metric("50년 학습 데이터", f"{cnt_50y}개년")
        st.markdown("<div class='metric-subtext'>1956~2005년 중 조건을 만족한 자료</div>", unsafe_allow_html=True)

    with c_t3:
        st.metric("100년 학습 데이터", f"{cnt_100y}개년")
        st.markdown("<div class='metric-subtext'>1906~2005년 중 조건을 만족한 자료</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box-blue">
        두 모델 모두 같은 최근 20년(2006~2025년)을 테스트 데이터로 사용합니다. 따라서 50년을 학습한 모델과 100년을 학습한 모델의 예측 성능을 공정하게 비교할 수 있습니다.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ==========================================
    # ② 전체 데이터로 만든 회귀모델
    # ==========================================
    st.markdown("### ② 전체 데이터로 만든 회귀모델")

    x_all = data['연도'].values
    y_all = data['연평균기온'].values

    # 회귀 모델 (전체 데이터)
    a_all, b_all = np.polyfit(x_all, y_all, 1)
    y_pred_all = a_all * x_all + b_all

    slope_100y_all = a_all * 100
    r_corr = np.corrcoef(x_all, y_all)[0, 1]
    mae_all = mean_absolute_error(y_all, y_pred_all)
    mse_all = mean_squared_error(y_all, y_pred_all)
    r2_all = r2_score(y_all, y_pred_all)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("100년당 기온 변화", f"{slope_100y_all:.2f} °C")
    c1.markdown(f"<div class='metric-subtext'>MSE = {mse_all:.3f}</div>", unsafe_allow_html=True)
    c2.metric("상관계수 r", f"{r_corr:.3f}")
    c3.metric("MAE", f"{mae_all:.3f} °C")
    c4.metric("R²", f"{r2_all:.3f}")

    st.markdown("""
    <div class="warning-box">
        이 평가는 전체 데이터를 이용해 회귀선을 만든 뒤 같은 데이터를 다시 평가한 결과입니다. 따라서 새로운 데이터에 대한 실제 예측 성능을 평가한 것은 아닙니다.
    </div>
    """, unsafe_allow_html=True)

    fig_all, ax_all = plt.subplots(figsize=(12, 4.8), facecolor='white')
    ax_all.set_facecolor('white')

    ax_all.scatter(x_all, y_all, color='#2563eb', s=25, label='실제 연평균기온', alpha=0.85)
    ax_all.plot(x_all, y_pred_all, color='#60a5fa', linewidth=2, label='전체데이터 회귀선')

    ax_all.set_ylabel("연평균기온 (°C)", fontsize=10, color='#666666')
    ax_all.set_xlabel("연도", fontsize=10, color='#666666')
    ax_all.set_ylim(9.0, 15.5)
    ax_all.grid(True, linestyle=':', alpha=0.4, color='#e5e7eb')

    for spine in ['top', 'right', 'left', 'bottom']:
        ax_all.spines[spine].set_color('#f3f4f6')

    ax_all.legend(loc='upper right', frameon=False, fontsize=9)
    st.pyplot(fig_all)

    st.markdown("---")

    # ==========================================
    # ③ 50년 학습과 100년 학습 비교
    # ==========================================
    st.markdown("### ③ 50년 학습과 100년 학습 비교")

    # 회귀 모델 학습
    a_50, b_50 = np.polyfit(train_50y['연도'], train_50y['연평균기온'], 1)
    a_100, b_100 = np.polyfit(train_100y['연도'], train_100y['연평균기온'], 1)

    # 테스트 예측 및 평가
    x_test = test_data['연도'].values
    y_test = test_data['연평균기온'].values

    y_pred_50 = a_50 * x_test + b_50
    y_pred_100 = a_100 * x_test + b_100

    mae_50 = mean_absolute_error(y_test, y_pred_50)
    mse_50 = mean_squared_error(y_test, y_pred_50)
    r2_50 = r2_score(y_test, y_pred_50)

    mae_100 = mean_absolute_error(y_test, y_pred_100)
    mse_100 = mean_squared_error(y_test, y_pred_100)
    r2_100 = r2_score(y_test, y_pred_100)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 최근 50년 학습")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("100년당 기온 변화", f"{a_50*100:.2f} °C")
        m2.metric("MAE", f"{mae_50:.3f} °C")
        m3.metric("MSE", f"{mse_50:.3f}")
        m4.metric("R²", f"{r2_50:.3f}")

    with col_right:
        st.markdown("#### 최근 100년 학습")
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("100년당 기온 변화", f"{a_100*100:.2f} °C")
        n2.metric("MAE", f"{mae_100:.3f} °C")
        n3.metric("MSE", f"{mse_100:.3f}")
        n4.metric("R²", f"{r2_100:.3f}")

    fig_comp, ax_comp = plt.subplots(figsize=(12, 4.8), facecolor='white')
    ax_comp.set_facecolor('white')

    ax_comp.axvspan(2006, 2025, color='#fee2e2', alpha=0.4, label='테스트 데이터 구간 (2006~2025)')

    train_past = data[data['연도'] <= 2005]
    ax_comp.scatter(train_past['연도'], train_past['연평균기온'], color='#94a3b8', s=25, alpha=0.6, label='과거 전체 데이터')
    ax_comp.
