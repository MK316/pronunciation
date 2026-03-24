import streamlit as st
import librosa
import numpy as np
import io
import plotly.graph_objects as go
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder

# --- 1. 페이지 및 초기화 설정 ---
st.set_page_config(page_title="Step 1: Rhythm Analyzer", layout="wide")

# 세션 상태 초기화: 녹음 데이터 존재 여부를 명확히 관리
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

def reset_all():
    # 모든 관련 세션 상태 초기화
    st.session_state.analysis_done = False
    for key in st.session_state.keys():
        if 'recorder' in key:
            del st.session_state[key]
    st.rerun()

st.title("📊 Step 1: Fluency & Rhythm Analyzer (v2.1)")
st.markdown("단어별로 뚝뚝 끊어 읽는 습관을 정밀 진단합니다.")

target_sentence = "The quick brown fox jumps over the lazy dog."
st.info(f"**Read this:** {target_sentence}")

# --- 2. 컨트롤 섹션 ---
col_rec, col_reset = st.columns([1, 5])
with col_rec:
    # 녹음기 (key를 고정하지 않고 매번 초기화할 수 있도록 구성 가능하지만 여기선 상태 체크 위주)
    audio_data = mic_recorder(
        start_prompt="🔴 Start Recording",
        stop_prompt="⏹️ Stop & Analyze",
        key='fluency_final'
    )

with col_reset:
    if audio_data or st.session_state.analysis_done:
        if st.button("🔄 Try Again / Reset"):
            reset_all()

# --- 3. 오디오 분석 및 결과 출력 ---
if audio_data:
    st.session_state.analysis_done = True
    try:
        with st.spinner("Analyzing rhythm patterns..."):
            audio_bytes = audio_data['bytes']
            
            # [수정] 너무 짧은 녹음(빈 파일) 방어 로직
            if len(audio_bytes) < 1000:
                st.warning("녹음된 내용이 너무 짧습니다. 다시 시도해 주세요.")
                st.stop()

            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)

            y, sr = librosa.load(wav_io, sr=None)
            
            # [수정] 무음만 있는 경우 체크
            if np.max(np.abs(y)) < 0.01:
                st.error("⚠️ 음성이 감지되지 않았습니다. 마이크 설정을 확인하거나 더 크게 말씀해 주세요.")
                st.stop()

            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
            total_duration = librosa.get_duration(y=y_trimmed, sr=sr)

            # --- 에너지 변화 분석 (Staccato 감지) ---
            rms = librosa.feature.rms(y=y_trimmed)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            frame_dur = librosa.get_duration(y=y_trimmed, sr=sr) / len(rms)

            # 임계값을 더 까다롭게 조정 (-25dB)
            threshold = -25 
            is_silent = rms_db < threshold
            
            silent_durations = []
            current_pause = 0
            for silent in is_silent:
                if silent: current_pause += frame_dur
                else:
                    if current_pause > 0: silent_durations.append(current_pause)
                    current_pause = 0

            # 0.05초~0.25초 사이의 미세 단절(Staccato)과 그 이상(Long Pause) 구분
            staccato_gaps = [d for d in silent_durations if 0.05 <= d < 0.25]
            long_pauses = [d for d in silent_durations if d >= 0.25]

            staccato_count = len(staccato_gaps)
            pause_count = len(long_pauses)
            total_pause_time = sum(staccato_gaps) + sum(long_pauses)
            actual_speech_time = max(0.1, total_duration - total_pause_time)
            phonation_ratio = (actual_speech_time / total_duration) * 100

            # 발화 속도 계산
            word_count = len(target_sentence.split())
            speech_rate = ((word_count * 1.3) / total_duration) * 60

        # --- 결과 대시보드 ---
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Speech Rate", f"{speech_rate:.1f} SPM")
        m2.metric("Long Pauses", f"{pause_count} 회")
        m3.metric("Staccato Gaps", f"{staccato_count} 회", delta=f"{staccato_count} detected", delta_color="inverse")
        m4.metric("Speech Density", f"{phonation_ratio:.1f} %")

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("⏱️ Speech Flow Visualization")
            fig = go.Figure(data=[go.Pie(
                labels=['Speaking', 'Long Pause', 'Staccato Gap'], 
                values=[actual_speech_time, sum(long_pauses), sum(staccato_gaps)],
                hole=.4, marker_colors=['#2E7D32', '#FFA000', '#D32F2F']
            )])
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("💡 Expert Feedback")
            # 피드백 우선순위: 아무것도 안 읽었을 때의 예외 처리
            if total_duration < 1.0:
                st.info("녹음 시간이 너무 짧습니다. 문장 전체를 읽어주세요.")
            elif staccato_count >= (word_count * 0.4):
                st.error("🚩 **Staccato Alert!**")
                st.write(f"단어 사이에서 {staccato_count}번의 끊김이 발생했습니다. 한국어식 음절 박자에서 벗어나 영어 특유의 **연음(Linking)**을 유지해 보세요.")
            elif phonation_ratio > 98:
                st.warning("쉼표나 마침표에서도 전혀 쉬지 않고 달리고 계시네요! 적절한 Chunking이 필요합니다.")
            else:
                st.success("자연스러운 연결성입니다. 훌륭합니다!")

    except Exception as e:
        st.error(f"분석 엔진 오류: {e}")

else:
    # 데이터가 없을 때 표시할 화면
    st.write("---")
    st.info("좌측 상단의 버튼을 눌러 녹음을 시작하세요. (최소 2초 이상 발화 권장)")
