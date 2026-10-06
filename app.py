import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import urllib.request
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit 페이지 설정
st.set_page_config(page_title="서울 기온 회귀 모델 평가 및 비교", layout="wide")

# --- 한글 폰트 자동 설정 (distutils 오류 방지) ---
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
        plt.rc('font', family='Nanum Gothic')
    else:
        font_names = [f.name for f in fm.fontManager.ttflist]
        if 'NanumGothic' in font_names:
            plt.rc('font', family='NanumGothic')
        elif 'Malgun Gothic' in font_names:
            plt.rc('font', family='Malgun Gothic')
        elif 'AppleGothic' in font_names:
            plt.rc('font', family='AppleGothic')
        else:
            plt.rc('font', family='DejaVu Sans')

    plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()

# 커스텀 CSS
st.markdown("""
<style>
    .stApp {
        background-color: #fcfbf7;
    }
    .main-card {
        background-color: #f7f5ed;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #e8e3d5;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌡️ 서울 연평균 기온 회귀 모델 학습 기간별 비교 평가")

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
    
    # 관측일수가 300일 이상인 해만 필터링
    valid_data = yearly_stats[yearly_stats['관측일수'] >= 300].copy()
    return valid_data

try:
    data = load_and_process_data()
    
    # --- 데이터셋 분할 ---
    test_data = data[(data['연도'] >= 2006) & (data['연도'] <= 2025)].copy()
    train_50y = data[(data['연도'] >= 1956) & (data['연도'] <= 2005)].copy()
    train_100y = data[(data['연도'] >= 1906) & (data['연도'] <= 2005)].copy()

    # --- 회귀 모델 학습 ---
    a_50, b_50 = np.polyfit(train_50y['연도'], train_50y['연평균기온'], 1)
    a_100, b_100 = np.polyfit(train_100y['연도'], train_100y['연평균기온'], 1)

    # --- 테스트 데이터(2006~2025) 예측 및 평가 ---
    x_test = test_data['연도'].values
    y_test = test_data['연평균기온'].values

    y_pred_50 = a_50 * x_test + b_50
    y_pred_100 = a_100 * x_test + b_100

    # 평가지표 계산
    mae_50 = mean_absolute_error(y_test, y_pred_50)
    mse_50 = mean_squared_error(y_test, y_pred_50)
    r2_50 = r2_score(y_test, y_pred_50)

    mae_100 = mean_absolute_error(y_test, y_pred_100)
    mse_100 = mean_squared_error(y_test, y_pred_100)
    r2_100 = r2_score(y_test, y_pred_100)

    # --- 카드 1: 요약 비교 지표 ---
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("📌 공통 테스트 데이터 (2006년~2025년) 모델 평가 결과")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("학습 기간", "최근 50년 (1956~2005)", f"기울기 +{a_50*100:.2f}°C/100년")
    col2.metric("MAE", f"{mae_50:.4f} °C", delta=f"{mae_50 - mae_100:.4f}", delta_color="inverse")
    col3.metric("MSE", f"{mse_50:.4f}", delta=f"{mse_50 - mse_100:.4f}", delta_color="inverse")
    col4.metric("R² 점수", f"{r2_50:.4f}", delta=f"{r2_50 - r2_100:.4f}")

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("학습 기간", "최근 100년 (1906~2005)", f"기울기 +{a_100*100:.2f}°C/100년")
    col2.metric("MAE", f"{mae_100:.4f} °C")
    col3.metric("MSE", f"{mse_100:.4f}")
    col4.metric("R² 점수", f"{r2_100:.4f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 카드 2: 회귀선 시각화 비교 ---
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("📈 회귀선 예측 결과 및 데이터 구간 시각화")

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#f7f5ed')
    ax.set_facecolor('#f7f5ed')

    # 테스트 구간 음영 (2006~2025)
    ax.axvspan(2006, 2025, color='#fee2e2', alpha=0.5, label='테스트 데이터 구간 (2006~2025)')

    # 데이터 산점도
    train_all = data[data['연도'] <= 2005]
    ax.scatter(train_all['연도'], train_all['연평균기온'], color='#94a3b8', s=30, alpha=0.6, label='과거 전체 관측 데이터')
    ax.scatter(test_data['연도'], test_data['연평균기온'], color='#ef4444', s=45, zorder=4, label='실제 테스트 데이터 (2006~2025)')

    # 회귀선
    x_range = np.linspace(1906, 2025, 200)
    ax.plot(x_range, a_50 * x_range + b_50, color='#2563eb', linewidth=2.5, linestyle='-', label=f'최근 50년 학습 회귀선 (+{a_50*100:.2f}°C/100년)')
    ax.plot(x_range, a_100 * x_range + b_100, color='#059669', linewidth=2.5, linestyle='--', label=f'최근 100년 학습 회귀선 (+{a_100*100:.2f}°C/100년)')

    ax.set_xlim(1900, 2028)
    ax.set_ylim(9.0, 15.5)
    ax.set_ylabel("°C", fontsize=11, rotation=0, loc='top')
    ax.set_xlabel("연도", fontsize=11, loc='right')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 카드 3: 평가 분석 보고서 ---
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("💡 기울기 및 예측 성능 평가 비교 분석")
    
    st.markdown(f"""
    1. **회귀선 기울기 비교**:
       - **최근 50년 학습 (1956~2005)**: 100년당 **+{a_50*100:.2f}°C** 상승 추세
       - **최근 100년 학습 (1906~2005)**: 100년당 **+{a_100*100:.2f}°C** 상승 추세
       - **인사이트**: 최근 50년 데이터로 학습한 모델의 기울기가 더 급격합니다. 이는 20세기 후반 이후 서울의 기온 상승 폭이 가속화되었음을 보여줍니다.

    2. **테스트 데이터 (2006~2025년) 예측 성능 비교**:
       - 최근 50년 데이터를 학습한 모델이 오차 지표(MAE, MSE)가 더 낮고 $R^2$ 점수가 높아 최근 기온 변화 흐름을 더 정확히 예측합니다.
       - 최근 100년 모델은 상대적으로 오래된 저기온 관측값이 포함되어 온난화 기울기가 완만하게 측정되므로, 최근 20년의 상승 경향을 낮게 예측하는 경향이 있습니다.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"데이터를 처리하는 중 오류가 발생했습니다: {e}")
