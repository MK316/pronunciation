import streamlit as st
import librosa
import numpy as np
import io
import plotly.graph_objects as go
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder

# --- 1. 페이지 설정 및 상태 초기화 ---
st.set_page_config(page_title="Step 1: Rhythm Analyzer", layout="wide")

# 리셋 로직: 모든 녹음 관련 상태를 완전히 비움
if st.sidebar.button("🔄 전체 초기화 (Reset All)"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

st.title("📊 Step 1: Fluency & Rhythm Analyzer")
st.markdown("음절 간의 연결성과 문장의 흐름을 분석하여 자연스러운 발화를 돕습니다.")

target_sentence = "The quick brown fox jumps over the lazy dog."
st.info(f"**Read this:** {target_sentence}")

# --- 2. 녹음 섹션 ---
# key에 session_state를 연동하여 리셋 시 버튼도 초기화되도록 유도
audio_data = mic_recorder(
    start_prompt="🔴 Start Recording",
    stop_prompt="⏹️ Stop & Analyze",
    key='rhythm_recorder_v3'
)

# --- 3. 분석 및 결과 출력 ---
if audio_data:
    try:
        with st.spinner("발화 리듬 분석 중..."):
            audio_bytes = audio_data['bytes']
            
            # 유효성 검사 (너무 짧은 소음 방지)
            if len(audio_bytes) < 2000:
                st.warning("녹음 시간이 너무 짧습니다. 문장 전체를 다시 읽어주세요.")
                if st.button("다시 시도"):
                    st.rerun()
                st.stop()

            # 오디오 로드 및 변환
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)

            y, sr = librosa.load(wav_io, sr=None)
            
            # Trimming (무음 제거)
            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
            total_duration = librosa.get_duration(y=y_trimmed, sr=sr)

            # --- 에너지 기반 리듬 분석 ---
            rms = librosa.feature.rms(y=y_trimmed)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            frame_dur = total_duration / len(rms)

            # 임계값 설정 (에너지가 뚝 떨어지는 지점 탐색)
            threshold = -25 
            is_silent = rms_db < threshold
            
            silent_durations = []
            current_pause = 0
            for silent in is_silent:
                if silent: current_pause += frame_dur
                else:
                    if current_pause > 0: silent_durations.append(current_pause)
                    current_pause = 0

            # 지표 분류: 0.05~0.25초(음절 간 끊김), 0.25초 이상(긴 멈춤)
            short_gaps = [d for d in silent_durations if 0.05 <= d < 0.25]
            long_pauses = [d for d in silent_durations if d >= 0.25]

            staccato_count = len(short_gaps)
            pause_count = len(long_pauses)
            total_pause_time = sum(short_gaps) + sum(long_pauses)
            actual_speech_time = max(0.1, total_duration - total_pause_time)
            phonation_ratio = (actual_speech_time / total_duration) * 100

            # 발화 속도 (SPM)
            word_count = len(target_sentence.split())
            speech_rate = ((word_count * 1.3) / total_duration) * 60

        # --- 4. 결과 대시보드 ---
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("발화 속도", f"{speech_rate:.1f} SPM")
        m2.metric("긴 멈춤", f"{pause_count} 회")
        m3.metric("음절 간 끊김", f"{staccato_count} 회", delta=f"{staccato_count}회 감지", delta_color="inverse")
        m4.metric("발화 밀도", f"{phonation_ratio:.1f} %")

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("⏱️ 시간 분포 (Speech vs Pause)")
            fig = go.Figure(data=[go.Pie(
                labels=['Speaking', 'Long Pause', 'Syllable Gap'], 
                values=[actual_speech_time, sum(long_pauses), sum(short_gaps)],
                hole=.4, marker_colors=['#2E7D32', '#FFA000', '#D32F2F']
            )])
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("💡 분석 피드백")
            if staccato_count >= (word_count * 0.4):
                st.error("🚩 **음절마다 끊어 읽기가 많아 보입니다.**")
                st.write("단어들을 하나씩 떼어서 읽기보다는, 소리를 다음 단어까지 부드럽게 밀어내는 **연음(Linking)** 연습을 해보세요. 한 호흡에 문장 끝까지 간다는 느낌이 중요합니다.")
            elif pause_count > 2:
                st.warning("문장 중간에 호흡이 자주 끊깁니다. 의미 단위(Chunk)로 묶어서 읽어보세요.")
            elif speech_rate < 100:
                st.info("전반적인 속도가 다소 느립니다. 조금 더 자신감 있게 읽어보세요.")
            else:
                st.success("단어 간 연결이 매끄럽고 리듬감이 아주 좋습니다!")

        # 하단에 '다시 시도' 버튼 배치 (스크롤 배려)
        if st.button("🔄 다시 시도하기 (Reset Analysis)"):
            reset_all()

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다. 다시 녹음해 주세요.")
        if st.button("오류 해결 후 다시 시도"):
            reset_all()

else:
    st.write("---")
    st.info("좌측 상단의 버튼을 눌러 녹음을 시작하세요. (문장 전체 발화 권장)")
