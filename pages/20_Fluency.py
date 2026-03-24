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
이 단계에서는 여러분의 **발화 속도(Speech Rate)**와 **휴지(Pause)** 패턴을 분석합니다. 
가급적 조용한 환경에서 아래 문장을 자연스럽게 읽어주세요.
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
        # 가이드: 데이터 로딩 중 메시지
        with st.spinner("Analyzing your voice..."):
            # A. 오디오 데이터 변환 (Any Format -> WAV)
            audio_bytes = audio_data['bytes']
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            # Librosa 분석을 위해 메모리 상에서 WAV로 추출
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)

            # B. Librosa로 오디오 로드
            y, sr = librosa.load(wav_io, sr=None)

            # [추가] 양 끝 무음 제거 (Trimming)
            # top_db=25: 25데시벨 이하를 무음으로 간주하고 앞뒤를 잘라냄
            y_trimmed, index = librosa.effects.trim(y, top_db=25)
            
            # 잘라낸 후의 실제 분석 대상 시간
            total_duration = librosa.get_duration(y=y_trimmed, sr=sr)

            # C. 유창성 지표 계산 (Pause Detection) - 잘라낸 오디오(y_trimmed) 사용
            non_silent_intervals = librosa.effects.split(y_trimmed, top_db=25)
            
            # 실제 발화 시간 합계 (초)
            actual_speech_time = sum([(end - start) / sr for start, end in non_silent_intervals])
            
            # 휴지 횟수 및 시간 (양 끝을 잘랐으므로 문장 내부의 휴지만 남음)
            pause_count = max(0, len(non_silent_intervals) - 1)
            pause_time = max(0, total_duration - actual_speech_time)

            # D. 발화 속도 계산 (Syllables Per Minute)
            # 영어교육적 관점에서 단어 수 * 1.5로 음절 수 대략적 추정
            word_count = len(target_sentence.split())
            estimated_syllables = word_count * 1.3 
            speech_rate = (estimated_syllables / total_duration) * 60  # SPM

        # --- 4. 시각화 결과 출력 ---
        st.divider()
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.metric("Speech Rate", f"{speech_rate:.1f} SPM")
            st.caption("원어민 평균: 130~150 SPM")
            
        with m_col2:
            st.metric("Total Pauses", f"{max(0, pause_count)} times")
            st.caption("의미 단위(Chunk)로 끊어 읽었는지 확인하세요.")

        with m_col3:
            st.metric("Speaking Ratio", f"{(actual_speech_time/total_duration)*100:.1f} %")
            st.caption("전체 녹음 중 실제 발화가 차지하는 비중")

        # 레이더 차트 또는 파이 차트 시각화
        c1, c2 = st.columns([2, 1])
        
        with c1:
            # 시간 분포 시각화 (Waveform & Silence)
            st.subheader("⏱️ Speech vs Silence Distribution")
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Actual Speaking', 'Silent Pauses'], 
                values=[actual_speech_time, pause_time],
                hole=.4,
                marker_colors=['#2E7D32', '#FFA000']
            )])
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.subheader("💡 Feedback")
            if speech_rate < 100:
                st.warning("발화 속도가 다소 느립니다. 단어 간의 연결(Linking)에 신경 쓰며 조금 더 속도를 높여보세요.")
            elif pause_count > (word_count / 2):
                st.error("불필요한 멈춤이 많습니다. 문장 구조를 미리 파악하고 호흡을 조절해 보세요.")
            else:
                st.success("훌륭합니다! 유창성과 리듬감이 안정적입니다.")

    except Exception as e:
        st.error(f"Error during analysis: {e}")
        st.info("녹음 시간이 너무 짧거나 마이크 설정에 문제가 있을 수 있습니다.")

else:
    st.write("---")
    st.info("왼쪽 상단의 'Start Recording' 버튼을 눌러 녹음을 시작하세요.")
