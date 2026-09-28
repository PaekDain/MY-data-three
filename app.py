import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import urllib.request
import os

# Streamlit 페이지 설정
st.set_page_config(page_title="서울 연평균 기온 예측기", layout="wide")

# --- 한글 폰트 자동 설정 (리눅스 / Streamlit Cloud / Windows / Mac 완벽 대응) ---
@st.cache_resource
def setup_korean_font():
    # 1. 나눔고딕 폰트 파일 다운로드 및 적용 (Streamlit Cloud 환경 대응)
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
        # 시스템 폰트 확인
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

# 커스텀 CSS (이미지 디자인 색상 및 부드러운 라운드 카드)
st.markdown("""
<style>
    .stApp {
        background-color: #fcfbf7;
    }
    .main-card {
        background-color: #f7f5ed;
        border-radius: 20px;
        padding: 24px;
        border: 1px solid #e8e3d5;
        box-shadow: 0 4px 16px rgba(0,0,0,0.02);
    }
</style>
""", unsafe_allow_html=True)

st.title("🌡️ 서울 연평균 기온 예측기")

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
    all_data = load_and_process_data()
    
    min_year_all = int(all_data['연도'].min())
    max_year_all = int(all_data['연도'].max())
    
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    # --- 회귀선 계산 (전체 데이터 기반 기본값) ---
    x_train = all_data['연도'].values
    y_train = all_data['연평균기온'].values
    a, b = np.polyfit(x_train, y_train, 1)
    slope_100y = a * 100

    # --- 그래프 그리기 ---
    fig, ax = plt.subplots(figsize=(12, 5.8), facecolor='#f7f5ed')
    ax.set_facecolor('#f7f5ed')

    # 1. 2025년 이후 "확인할 데이터 없음" 미래 영역 음영 처리
    ax.axvspan(2025, 2100, color='#eeebe3', alpha=0.8, zorder=1)
    ax.text(2070, 10.0, "확인할\n데이터 없음", fontsize=11, color='#888479', 
            ha='center', va='center', zorder=2, multialignment='center')

    # 2. 실선 회귀선 (1908~2025) 및 점선 외삽선 (2025~2100)
    x_fit = np.linspace(min_year_all, 2025, 100)
    y_fit = a * x_fit + b
    ax.plot(x_fit, y_fit, color='#3b82f6', linewidth=2.2, zorder=3)

    x_extrap = np.linspace(2025, 2100, 100)
    y_extrap = a * x_extrap + b
    ax.plot(x_extrap, y_extrap, color='#3b82f6', linestyle='--', linewidth=2, zorder=3)

    # 3. 데이터 산점도 (학습 데이터)
    ax.scatter(all_data['연도'], all_data['연평균기온'], 
               color='#70a5e6', s=35, alpha=0.85, zorder=4)

    # --- 컨트롤 영역 (슬라이더 및 빠르게 고르기) ---
    col_slider, col_btn1, col_btn2, col_btn3, col_result = st.columns([4, 1.2, 1.2, 1.2, 3])
    
    # 기본값 설정
    if "target_year" not in st.session_state:
        st.session_state.target_year = 2045

    with col_btn1:
        if st.button("2025년", use_container_width=True):
            st.session_state.target_year = 2025
    with col_btn2:
        if st.button("2045년", use_container_width=True):
            st.session_state.target_year = 2045
    with col_btn3:
        if st.button("2100년", use_container_width=True):
            st.session_state.target_year = 2100

    with col_slider:
        target_year = st.slider(
            "연도",
            min_value=1900,
            max_value=2100,
            value=st.session_state.target_year,
            key="slider_year"
        )
        st.session_state.target_year = target_year

    predicted_temp = a * target_year + b

    # 4. 예측 수직 점선 & 선택 포인트 (빨간색 포인트)
    ax.axvline(target_year, color='#888888', linestyle=':', linewidth=1.5, zorder=5)
    ax.scatter([target_year], [predicted_temp], color='#ef4444', s=50, zorder=6)

    # 5. 상단 굵은 예측값 타이틀
    ax.text(target_year, 17.0, f"{target_year}년 예측 {predicted_temp:.1f}°C", 
            fontsize=13, fontweight='bold', color='#111111', ha='center', va='bottom', zorder=6)

    # 6. 우측 상단 "외삽" 빨간색 배지 (2025년 초과시 표시)
    if target_year > 2025:
        ax.text(2090, 17.2, " 외삽 ", fontsize=11, fontweight='bold', color='white',
                ha='center', va='center', zorder=7,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#ef4444', edgecolor='none'))

    # 7. 그래프 좌측 하단 메타 정보
    info_sub = f"회귀선 · 학습 데이터 {min_year_all}~{max_year_all} · 기울기 +{slope_100y:.2f}°C/100년"
    ax.text(1900, 8.2, info_sub, fontsize=10.5, color='#666666', ha='left', va='top')

    # 축 범위 및 스타일 조정
    ax.set_ylim(9.0, 18.0)
    ax.set_xlim(1895, 2105)
    ax.set_ylabel("°C", fontsize=11, rotation=0, loc='top', color='#666666')
    ax.set_xlabel("연도", fontsize=10.5, loc='right', color='#666666')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.grid(True, linestyle=':', alpha=0.5, color='#d5d1c5')

    st.pyplot(fig)

    # 우측 하단 예측 결과 강조 출력
    with col_result:
        st.markdown(
            f"<h4 style='text-align: right; color: #d97706; margin-top: 10px;'>"
            f"{target_year}년 ➔ 예측값 {predicted_temp:.1f}°C"
            f"</h4>",
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
