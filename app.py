import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# Streamlit 페이지 설정
st.set_page_config(page_title="서울 연평균 기온 예측기", layout="wide")

# Matplotlib 한글 폰트 설정
@st.cache_resource
def setup_font():
    # 리눅스/윈도우/맥 한글 폰트 설정
    font_list = [f.name for f in fm.fontManager.ttflist]
    if 'NanumGothic' in font_list:
        plt.rc('font', family='NanumGothic')
    elif 'Malgun Gothic' in font_list:
        plt.rc('font', family='Malgun Gothic')
    elif 'AppleGothic' in font_list:
        plt.rc('font', family='AppleGothic')
    else:
        # 폰트가 없는 경우를 대비한 시스템 폰트 설정
        plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

setup_font()

# 커스텀 CSS (이미지 스타일의 아이보리 톤 및 카드 디자인 반영)
st.markdown("""
<style>
    .stApp {
        background-color: #f7f5ed;
    }
    .main-card {
        background-color: #f3efe0;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #e2dac7;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
</style>
""", unsafe_allow_html=unsafe_allow_html)

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
    
    # 연도별 관측일수 및 연평균기온 계산
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
    
    # --- 상단 설정 및 컨트롤 영역 ---
    col_preset, col_slider, col_target = st.columns([2, 3, 2])
    
    with col_preset:
        st.write("**학습 시작 연도 빠른 선택**")
        preset = st.radio(
            "학습 범위 버튼",
            ["전체", "최근 50년", "최근 30년", "최근 20년"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if preset == "전체":
            default_start = min_year_all
        elif preset == "최근 50년":
            default_start = max_year_all - 50 + 1
        elif preset == "최근 30년":
            default_start = max_year_all - 30 + 1
        elif preset == "최근 20년":
            default_start = max_year_all - 20 + 1

    with col_slider:
        start_year = st.slider(
            "학습 시작 연도",
            min_value=min_year_all,
            max_value=max_year_all - 5,
            value=default_start
        )

    with col_target:
        target_year = st.slider(
            "예측 대상 연도",
            min_value=1900,
            max_value=2100,
            value=2045
        )

    # --- 데이터 분리 및 회귀 계산 ---
    train_data = all_data[all_data['연도'] >= start_year]
    excluded_data = all_data[all_data['연도'] < start_year]
    
    num_train_years = len(train_data)
    
    # 회귀 계산 (y = ax + b)
    x_train = train_data['연도'].values
    y_train = train_data['연평균기온'].values
    a, b = np.polyfit(x_train, y_train, 1)
    
    # 100년당 기온 상승률 (°C/100년)
    slope_100y = a * 100
    
    # 예측 연도의 기온
    predicted_temp = a * target_year + b
    
    # --- 그래프 그리기 (이미지 디자인 스타일 재현) ---
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#f3efe0')
    ax.set_facecolor('#f3efe0')
    
    # 배경 영역 표시 (학습 구간)
    ax.axvspan(start_year, max_year_all, color='#e5f0f7', alpha=0.5, zorder=1)
    
    # 1. 학습 제외 데이터
    if len(excluded_data) > 0:
        ax.scatter(excluded_data['연도'], excluded_data['연평균기온'], 
                   color='#b0bec5', s=30, alpha=0.7, label='학습 제외', zorder=2)
        
    # 2. 학습 데이터
    ax.scatter(train_data['연도'], train_data['연평균기온'], 
               color='#64b5f6', s=45, alpha=0.9, label='학습 데이터', zorder=3)
    
    # 3. 실선 회귀선 (학습 구간)
    x_fit = np.linspace(start_year, max_year_all, 100)
    y_fit = a * x_fit + b
    ax.plot(x_fit, y_fit, color='#1976d2', linewidth=2.5, label='회귀선', zorder=4)
    
    # 4. 점선 외삽선 (예측 연도까지)
    if target_year > max_year_all:
        x_extrap = np.linspace(max_year_all, target_year, 100)
        y_extrap = a * x_extrap + b
        ax.plot(x_extrap, y_extrap, color='#1976d2', linestyle='--', linewidth=2, label='외삽', zorder=4)
    elif target_year < start_year:
        x_extrap = np.linspace(target_year, start_year, 100)
        y_extrap = a * x_extrap + b
        ax.plot(x_extrap, y_extrap, color='#1976d2', linestyle='--', linewidth=2, label='외삽', zorder=4)

    # 5. 예측 지점 강조 표시 (원형 마커)
    ax.scatter([target_year], [predicted_temp], facecolors='none', edgecolors='#af7a15', 
               s=120, linewidth=2.5, zorder=5)
    ax.text(target_year, predicted_temp + 0.35, f"{predicted_temp:.1f}°C", 
            color='#af7a15', fontweight='bold', fontsize=11, ha='center')

    # 6. 상단 요약 툴팁 박스 (이미지의 정보 창)
    info_text = (
        f"학습 {num_train_years}개 해 ({start_year}~{max_year_all})\n"
        f"기울기 +{slope_100y:.2f}°C/100년\n"
        f"{target_year}년 {predicted_temp:.1f}°C"
    )
    
    # 박스 위치 설정 (그래프 상단 중앙)
    mid_x = (min_year_all + max(target_year, max_year_all)) / 2
    max_y = max(all_data['연평균기온'].max(), predicted_temp) + 1.2
    
    ax.text(mid_x, max_y - 0.5, info_text, fontsize=11, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='#e0e0e0', alpha=0.95),
            color='#333333', linespacing=1.5)

    # 축 스타일링
    ax.set_ylabel("°C", fontsize=12, rotation=0, loc='top', color='#555555')
    ax.set_xlabel("연도", fontsize=11, loc='right', color='#555555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#b0bec5')
    ax.spines['bottom'].set_color('#b0bec5')
    ax.grid(True, linestyle=':', alpha=0.5, color='#cccccc')
    
    # 범례 설정
    ax.legend(loc='lower right', frameon=False, fontsize=10, ncol=4)
    
    st.pyplot(fig)
    
    # 하단 텍스트 정보 표시
    st.markdown(
        f"<p style='text-align: right; color: #8d6e63; font-weight: bold; font-size: 15px;'>"
        f"학습 {start_year}~{max_year_all} · 기울기 +{slope_100y:.2f}°C/100년"
        f"</p>", 
        unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
