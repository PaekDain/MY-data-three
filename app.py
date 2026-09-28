import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit 페이지 설정
st.set_page_config(page_title="기온 예측기", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")

# 데이터 URL 정의
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_process_data():
    # CSV 데이터 로드 (UTF-8 인코딩)
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    
    # 열 이름 정리 (앞뒤 공백 제거)
    df.columns = df.columns.str.strip()
    
    # 날짜 데이터형 변환 및 연도 추출
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    # 2025년 이하 데이터만 필터링
    df = df[df['연도'] <= 2025]
    
    # 연도별 관측일수 및 평균기온 계산
    yearly_stats = df.groupby('연도').agg(
        관측일수=('평균기온', 'count'),
        연평균기온=('평균기온', 'mean')
    ).reset_index()
    
    # 관측일수가 300일 이상인 해만 필터링
    valid_data = yearly_stats[yearly_stats['관측일수'] >= 300].copy()
    
    return valid_data

try:
    data = load_and_process_data()
    
    # 데이터 관련 정보 계산
    start_year = int(data['연도'].min())
    end_year = int(data['연도'].max())
    num_years = len(data)
    
    # 선형 회귀 직선 계산 (y = ax + b)
    x = data['연도'].values
    y = data['연평균기온'].values
    a, b = np.polyfit(x, y, 1) # a: 기울기, b: 절편
    
    # 상관계수 계산
    corr = np.corrcoef(x, y)[0, 1]
    
    # 요약 정보 표시
    st.subheader("📊 데이터 요약 정보")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("분석 대상 시작 연도", f"{start_year}년")
    col2.metric("분석 대상 끝 연도", f"{end_year}년")
    col3.metric("직선을 만든 해의 개수", f"{num_years}개")
    col4.metric("연도와 기온의 상관계수", f"{corr:.4f}")

    st.markdown("---")

    # 슬라이더를 통한 연도 선택 및 예상 기온 표시
    st.subheader("🔮 연도별 예상 기온 예측")
    selected_year = st.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2025)
    
    predicted_temp = a * selected_year + b
    
    st.metric(
        label=f"👉 {selected_year}년 예상 연평균 기온",
        value=f"{predicted_temp:.2f} °C"
    )

    st.markdown("---")

    # 산점도 및 회귀 직선 시각화
    st.subheader("📈 연도별 연평균 기온 산점도 및 회귀 직선")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 산점도 그리기
    ax.scatter(data['연도'], data['연평균기온'], color='skyblue', edgecolors='blue', alpha=0.7, label='실제 연평균기온')
    
    # 회귀 직선 그리기 (1900년~2100년 범위 전체 추세선)
    x_range = np.linspace(1900, 2100, 200)
    y_range = a * x_range + b
    ax.plot(x_range, y_range, color='red', linestyle='--', linewidth=2, label=f'회귀 직선 (y = {a:.4f}x + {b:.2f})')
    
    # 선택된 연도 강조 표시
    ax.scatter([selected_year], [predicted_temp], color='green', s=150, zorder=5, label=f'선택된 연도 ({selected_year}년)')
    
    ax.set_title("Seoul Annual Average Temperature Trend", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Average Temperature (°C)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    
    st.pyplot(fig)

except Exception as e:
    st.error(f"데이터를 불러오거나 처리하는 중 오류가 발생했습니다: {e}")
