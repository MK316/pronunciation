import streamlit as st
import librosa
import numpy as np
import io
import plotly.graph_objects as go
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="Step 1: Fluency Analyzer", layout="wide")

st.title("📊 Step 1: Fluency & Rhythm Analyzer")
st.markdown("""
이 단계에서는 여러분의 **발화 속도(Speech Rate)**와 **유의미한 휴지(Pause)** 패턴을 분석합니다. 
단어 사이의 미세한 끊김이 아닌, 실제 '멈춤'을 측정합니다.
""")

# 분석 대상 예시 문장
target_sentence = "The quick brown fox jumps over the lazy dog."
st.info(f"**Read this:** {target_sentence}")

# --- 2. 음성 녹음 섹션 ---
col_rec, col_empty = st.columns([1, 2])
with col_rec:
    audio_data = mic_recorder(
        start_prompt="🔴 Start Recording",
        stop_prompt="⏹️ Stop & Analyze",
        key='fluency_recorder'
    )

# --- 3. 오디오 분석 로직 ---
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
            # top_db=30: 30데시벨 이하를 무음으로 간주하고 앞뒤 공백 제거
            y_trimmed, index = librosa.effects.trim(y, top_db=30)
            total_duration = librosa.get_duration(y=y_trimmed, sr=sr)

            # C. 유의미한 멈춤(Pause) 필터링 로직
            # 30dB 이하 구간을 나누되, 아주 짧은 구간은 무시함
            intervals = librosa.effects.split(y_trimmed, top_db=30)
            
            # 언어학적 기준: 0.3초(300ms) 이상의 공백만 실제 Pause로 인정
            min_pause_duration = 0.3 
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

            # D. 발화 속도 계산 (Syllables Per Minute)
            word_count = len(target_sentence.split())
            estimated_syllables = word_count * 1.3 # 영어 평균 음절 가중치
            speech_rate = (estimated_syllables / total_duration) * 60  # SPM

        # --- 4. 시각화 결과 출력 ---
        st.divider()
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.metric("Speech Rate", f"{speech_rate:.1f} SPM")
            st.caption("원어민 평균: 130~150 SPM")
            
        with m_col2:
            st.metric("Significant Pauses", f"{pause_count} times")
            st.caption(f"{min_pause_duration}초 이상의 멈춤만 카운트됨")

        with m_col3:
            st.metric("Speaking Ratio", f"{(actual_speech_time/total_duration)*100:.1f} %")
            st.caption("순수 발화 시간 비중")

        # 결과 시각화
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("⏱️ Speech vs Pause Distribution")
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Actual Speaking', 'Long Pauses'], 
                values=[actual_speech_time, total_pause_time],
                hole=.4,
                marker_colors=['#2E7D32', '#FFA000']
            )])
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.subheader("💡 Analysis Feedback")
            if speech_rate < 100:
                st.warning("문장을 읽는 속도가 다소 느립니다. 단어들을 좀 더 매끄럽게 연결해 보세요.")
            elif pause_count > 2:
                st.error(f"문장 내에서 {pause_count}번의 뚜렷한 멈춤이 감지되었습니다. 의미 단위로 호흡을 조절해 보세요.")
            else:
                st.success("훌륭합니다! 불필요한 멈춤 없이 유창하게 발화하셨습니다.")

    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")
        st.info("녹음 데이터가 너무 짧거나 환경음에 문제가 있을 수 있습니다.")

else:
    st.write("---")
    st.info("녹음 버튼을 누르고 문장을 읽어주세요. 분석 결과가 여기에 표시됩니다.")
