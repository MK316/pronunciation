import streamlit as st
import librosa
import numpy as np
import io
import plotly.graph_objects as go
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="Step 1: Fluency & Rhythm Analyzer", layout="wide")

def reset_analysis():
    for key in st.session_state.keys():
        if 'recorder' in key:
            del st.session_state[key]
    st.rerun()

st.title("📊 Step 1: Fluency & Rhythm Analyzer")
st.markdown("""
이 도구는 **발화 속도**와 **문장의 흐름**을 분석합니다.  
단어 하나하나를 떼어 읽지 않고, 의미 단위로 부드럽게 이어서 읽는 것이 중요합니다.
""")

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

if audio_data and st.button("🔄 Try Again"):
    reset_analysis()

# --- 3. 오디오 분석 로직 ---
if audio_data:
    try:
        with st.spinner("Analyzing rhythm and flow..."):
            # A. 오디오 데이터 변환 (WAV)
            audio_bytes = audio_data['bytes']
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)

            # B. Librosa로 로드 및 양 끝 무음 제거
            y, sr = librosa.load(wav_io, sr=None)
            y_trimmed, _ = librosa.effects.trim(y, top_db=30)
            total_duration = librosa.get_duration(y=y_trimmed, sr=sr)

            # C. 이중 멈춤 감지 로직 (Two-tier Pause Detection)
            # top_db를 35로 높여 미세한 소리 단절에 더 예민하게 반응하도록 설정
            intervals = librosa.effects.split(y_trimmed, top_db=35)
            
            long_pauses = []   # 0.3초 이상: 의미적 단절
            short_gaps = []    # 0.1초 ~ 0.3초: 단어별 끊어 읽기 (Staccato)
            
            for i in range(len(intervals) - 1):
                gap_dur = (intervals[i+1][0] - intervals[i][1]) / sr
                
                if gap_dur >= 0.3:
                    long_pauses.append(gap_dur)
                elif gap_dur >= 0.1:
                    short_gaps.append(gap_dur)

            pause_count = len(long_pauses)
            staccato_count = len(short_gaps)
            total_pause_time = sum(long_pauses) + sum(short_gaps)
            actual_speech_time = max(0.1, total_duration - total_pause_time)

            # D. 발화 속도 계산
            word_count = len(target_sentence.split())
            estimated_syllables = word_count * 1.3 
            speech_rate = (estimated_syllables / total_duration) * 60  

        # --- 4. 시각화 결과 출력 ---
        st.divider()
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            st.metric("Speech Rate", f"{speech_rate:.1f} SPM")
            st.caption("Native: 130~150 SPM")
            
        with m_col2:
            st.metric("Long Pauses", f"{pause_count} 회")
            st.caption("0.3초 이상 멈춤")

        with m_col3:
            # 단어별 끊어 읽기가 많을수록 이 수치가 올라감
            st.metric("Staccato Gaps", f"{staccato_count} 회", delta=staccato_count, delta_color="inverse")
            st.caption("단어 사이 미세 단절")

        with m_col4:
            phonation_ratio = (actual_speech_time / total_duration) * 100
            st.metric("Speech Density", f"{phonation_ratio:.1f} %")
            st.caption("발화의 밀도")

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("⏱️ Speech Flow Analysis")
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Actual Speaking', 'Long Pauses', 'Staccato Gaps'], 
                values=[actual_speech_time, sum(long_pauses), sum(short_gaps)],
                hole=.4,
                marker_colors=['#2E7D32', '#FFA000', '#F44336']
            )])
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.subheader("💡 Expert Feedback")
            # 피드백 로직 고도화
            if staccato_count > (word_count * 0.4):
                st.error("🚩 **Staccato Alert!**")
                st.write("단어들을 너무 뚝뚝 끊어서 읽고 있습니다. 영어는 단어와 단어가 연결되는 **Linking(연음)**이 중요합니다. 숨을 멈추지 말고 문장 끝까지 밀어내듯 발음해 보세요.")
            elif speech_rate < 110:
                st.warning("발화 속도가 다소 느립니다. 조금 더 자신감 있게 속도를 높여보세요.")
            elif pause_count > 1:
                st.info("문장 중간에 호흡이 끊기는 구간이 있습니다. 의미 단위(Meaning Chunk)를 확인해 보세요.")
            else:
                st.success("훌륭합니다! 단어 간 연결이 매끄럽고 리듬감이 원어민과 유사합니다.")

    except Exception as e:
        st.error(f"Analysis Error: {e}")
else:
    st.write("---")
    st.info("좌측 상단의 버튼을 눌러 녹음을 시작하세요.")
