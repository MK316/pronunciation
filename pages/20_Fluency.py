import streamlit as st
import librosa
import numpy as np
import io
import plotly.graph_objects as go
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="Step 1: Fluency Analyzer", layout="wide")

# 세션 상태 초기화 함수
def reset_analysis():
    for key in st.session_state.keys():
        if 'recorder' in key:
            del st.session_state[key]
    st.rerun()

st.title("📊 Step 1: Fluency & Rhythm Analyzer")
st.markdown("""
이 단계에서는 여러분의 **발화 속도(Speech Rate)**와 **유의미한 휴지(Pause)** 패턴을 분석합니다. 
""")

# 분석 대상 예시 문장
target_sentence = "The quick brown fox jumps over the lazy dog."
st.info(f"**Read this:** {target_sentence}")

# --- 2. 음성 녹음 섹션 ---
col_rec, col_reset = st.columns([1, 5])
with col_rec:
    audio_data = mic_recorder(
        start_prompt="🔴 Start Recording",
        stop_prompt="⏹️ Stop & Analyze",
        key='fluency_recorder'
    )

with col_reset:
    if audio_data:
        # 다시 시도 버튼: 클릭 시 모든 데이터 초기화 후 새로고침
        if st.button("🔄 Try Again"):
            reset_analysis()

# --- 3. 오디오 분석 및 결과 출력 ---
if audio_data:
    try:
        with st.spinner("Analyzing your fluency..."):
            # A. 오디오 데이터 변환 (WAV)
            audio_bytes = audio_data['bytes']
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)

            # B. Librosa로 로드 및 양 끝 무음 제거 (Trimming)
            y, sr = librosa.load(wav_io, sr=None)
            y_trimmed, _ = librosa.effects.trim(y, top_db=30)
            total_duration = librosa.get_duration(y=y_trimmed, sr=sr)

            # C. 유의미한 멈춤(Pause) 필터링 (0.3초 기준)
            intervals = librosa.effects.split(y_trimmed, top_db=30)
            min_pause_duration = 0.2 
            valid_pauses = []
            total_pause_time = 0
            
            for i in range(len(intervals) - 1):
                pause_start = intervals[i][1] / sr
                pause_end = intervals[i+1][0] / sr
                pause_dur = pause_end - pause_start
                if pause_dur >= min_pause_duration:
                    valid_pauses.append((pause_start, pause_end))
                    total_pause_time += pause_dur

            pause_count = len(valid_pauses)
            actual_speech_time = max(0.1, total_duration - total_pause_time)

            # D. 발화 속도 계산
            word_count = len(target_sentence.split())
            estimated_syllables = word_count * 1.3 
            speech_rate = (estimated_syllables / total_duration) * 60  

        # --- 4. 시각화 결과 출력 (데이터가 있을 때만 표시) ---
        st.divider()
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.metric("Speech Rate", f"{speech_rate:.1f} SPM")
            st.caption("Native Speaker: 130~150 SPM")
            
        with m_col2:
            st.metric("Significant Pauses", f"{pause_count} times")
            st.caption(f"Paused for more than {min_pause_duration}s")

        with m_col3:
            st.metric("Speaking Ratio", f"{(actual_speech_time/total_duration)*100:.1f} %")
            st.caption("Pure Speaking Time Ratio")

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("⏱️ Speech vs Pause Distribution")
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Speaking', 'Long Pauses'], 
                values=[actual_speech_time, total_pause_time],
                hole=.4,
                marker_colors=['#2E7D32', '#FFA000']
            )])
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.subheader("💡 Feedback")
            if speech_rate < 100:
                st.warning("발화 속도가 느립니다. Linking(연음)에 신경 써보세요.")
            elif pause_count > 2:
                st.error(f"{pause_count}번의 유의미한 멈춤이 감지되었습니다.")
            else:
                st.success("훌륭합니다! 리듬감이 매우 좋습니다.")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.write("---")
    st.info("좌측 상단의 버튼을 눌러 녹음을 시작하세요.")
