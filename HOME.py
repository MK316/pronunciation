import streamlit as st
from array import array
import librosa
import numpy as np
import plotly.graph_objects as go
from streamlit_mic_recorder import mic_recorder
import io

st.set_page_config(page_title="Step 1: Fluency Analyzer", layout="wide")

st.title("📊 Step 1: Fluency & Rhythm Analyzer")
st.markdown("""
이 단계에서는 여러분의 **발화 속도**와 **끊어 읽기(Pause)**를 분석합니다. 
준비가 되면 아래 녹음 버튼을 누르고 주어진 문장을 읽어주세요.
""")

# 분석 대상 문장 예시
target_sentence = "The quick brown fox jumps over the lazy dog."
st.info(f"**Read this:** {target_sentence}")

# --- 음성 녹음 섹션 ---
audio_data = mic_recorder(
    start_prompt="🔴 Start Recording",
    stop_prompt="⏹️ Stop & Analyze",
    key='recorder'
)

if audio_data:
    # 1. 오디오 로드 (BytesIO -> Librosa)
    audio_bytes = io.BytesIO(audio_data['bytes'])
    y, sr = librosa.load(audio_bytes, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # 2. 휴지(Pause) 분석
    # 데시벨 기준(top_db) 이하인 구간을 무음으로 간주
    non_silent_intervals = librosa.effects.split(y, top_db=30)
    
    # 발화 시간 계산
    actual_speech_time = sum([(end - start) / sr for start, end in non_silent_intervals])
    pause_count = len(non_silent_intervals) - 1
    pause_time = duration - actual_speech_time

    # 3. 유창성 지표 계산 (예시 음절 수 기반)
    # 실제 구현시에는 ASR로 음절수를 세거나, 문장 길이를 기준으로 고정 가능
    estimated_syllables = len(target_sentence.split()) * 1.5 # 단순 추정치
    speech_rate = (estimated_syllables / duration) * 60 # SPM (Syllables Per Minute)

    # --- 시각화 섹션 ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Fluency Metrics")
        st.metric("Speech Rate", f"{speech_rate:.1f} SPM")
        st.metric("Total Pauses", f"{max(0, pause_count)} times")
        st.progress(min(speech_rate / 150, 1.0), text="Speech Speed Intensity")

    with col2:
        st.subheader("⏱️ Time Breakdown")
        # 파이 차트로 발화 시간 vs 휴지 시간 비율 표시
        fig = go.Figure(data=[go.Pie(labels=['Speaking', 'Pausing'], 
                             values=[actual_speech_time, pause_time],
                             hole=.3,
                             marker_colors=['#4CAF50', '#FFC107'])])
        st.plotly_chart(fig, use_container_width=True)

    # 4. 피드백 메시지
    st.divider()
    if speech_rate < 100:
        st.warning("⚠️ 조금 더 빠르게 읽어보세요. 원어민 평균은 분당 130-150음절입니다.")
    elif pause_count > 3:
        st.info("ℹ️ 문장 중간에 끊어 읽기가 잦습니다. 의미 단위(Meaning Chunk)로 이어 읽는 연습이 필요합니다.")
    else:
        st.success("✅ 유창성이 매우 훌륭합니다! 리듬감이 안정적입니다.")
