import streamlit as st
import librosa
import numpy as np
import io
import plotly.graph_objects as go
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder

st.set_page_config(page_title="Step 1: Advanced Rhythm Analyzer", layout="wide")

def reset_analysis():
    for key in st.session_state.keys():
        if 'recorder' in key:
            del st.session_state[key]
    st.rerun()

st.title("📊 Step 1: Fluency & Rhythm Analyzer (v2.0)")
st.markdown("단어별로 뚝뚝 끊어 읽는 습관을 정밀하게 진단합니다.")

target_sentence = "The quick brown fox jumps over the lazy dog."
st.info(f"**Read this:** {target_sentence}")

col_rec, col_reset = st.columns([1, 5])
with col_rec:
    audio_data = mic_recorder(start_prompt="🔴 Start Recording", stop_prompt="⏹️ Stop & Analyze", key='fluency_v2')
if audio_data and st.button("🔄 Try Again"):
    reset_analysis()

if audio_data:
    try:
        with st.spinner("Analyzing rhythm patterns..."):
            # A. 데이터 로드 및 전처리
            audio_bytes = audio_data['bytes']
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)

            y, sr = librosa.load(wav_io, sr=None)
            # 더 공격적인 Trimming (앞뒤 공백 확실히 제거)
            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
            total_duration = librosa.get_duration(y=y_trimmed, sr=sr)

            # B. [핵심] 에너지 변화를 통한 단어 끊김 감지
            # 단순히 데시벨만 보는게 아니라 소리의 에너지가 뚝 떨어지는 지점을 찾음
            rms = librosa.feature.rms(y=y_trimmed)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            times = librosa.frames_to_time(range(len(rms)), sr=sr)

            # 단어별 끊기 감지를 위한 임계값 (더 까다롭게 설정)
            # 에너지가 최고점 대비 -20dB 이하로 떨어지는 구간을 탐색
            threshold = -20 
            is_silent = rms_db < threshold
            
            # 무음 구간의 길이 계산 로직
            silent_durations = []
            current_pause = 0
            frame_dur = times[1] - times[0]

            for silent in is_silent:
                if silent:
                    current_pause += frame_dur
                else:
                    if current_pause > 0:
                        silent_durations.append(current_pause)
                    current_pause = 0

            # C. 지표 재정의
            # 0.05초(50ms) 이상만 되어도 단어 간 단절로 간주 (한국식 끊어읽기 타겟)
            staccato_gaps = [d for d in silent_durations if 0.05 <= d < 0.3]
            long_pauses = [d for d in silent_durations if d >= 0.3]

            staccato_count = len(staccato_gaps)
            pause_count = len(long_pauses)
            total_pause_time = sum(staccato_gaps) + sum(long_pauses)
            actual_speech_time = max(0.1, total_duration - total_pause_time)
            phonation_ratio = (actual_speech_time / total_duration) * 100

            # D. 발화 속도 계산
            word_count = len(target_sentence.split())
            speech_rate = ((word_count * 1.3) / total_duration) * 60

        # --- 결과 출력 ---
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Speech Rate", f"{speech_rate:.1f} SPM")
        m2.metric("Long Pauses", f"{pause_count} 회")
        # delta를 통해 끊어 읽기가 많을 때 시각적 경고 강조
        m3.metric("Staccato Gaps", f"{staccato_count} 회", delta=f"{staccato_count} detected", delta_color="inverse")
        m4.metric("Speech Density", f"{phonation_ratio:.1f} %")

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("⏱️ Speech Density Chart")
            # 에너지가 낮아진 구간을 더 명확히 시각화
            fig = go.Figure(data=[go.Pie(
                labels=['Speaking', 'Long Pause', 'Staccato Gap'], 
                values=[actual_speech_time, sum(long_pauses), sum(short_gaps) if 'short_gaps' in locals() else sum(staccato_gaps)],
                hole=.4, marker_colors=['#2E7D32', '#FFA000', '#D32F2F']
            )])
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("💡 Expert Diagnosis")
            if staccato_count >= (word_count * 0.5):
                st.error("🚩 **한국식 단어별 끊어 읽기 감지!**")
                st.write(f"현재 {staccato_count}번의 미세한 단절이 발견되었습니다. 단어 끝을 길게 끌거나 멈추지 말고, 다음 단어로 소리를 '던지듯' 연결하세요.")
            elif phonation_ratio > 95 and speech_rate > 160:
                st.warning("너무 급하게 읽고 계신가요? 억양의 고저 없이 밀어붙이는 발화일 수 있습니다.")
            else:
                st.success("자연스러운 흐름입니다.")

    except Exception as e:
        st.error(f"Error: {e}")
