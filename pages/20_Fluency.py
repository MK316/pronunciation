import streamlit as st
import librosa
import numpy as np
import io
import plotly.graph_objects as go
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS

# --- 1. 페이지 설정 및 상태 초기화 ---
st.set_page_config(page_title="Step 1: Rhythm Analyzer", layout="wide")

# 세션 상태 관리: 분석 결과 유무를 명확히 제어
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# 리셋 함수: 모든 상태를 초기화하고 페이지를 새로고침
def reset_app():
    st.session_state.analysis_result = None
    for key in list(st.session_state.keys()):
        if 'recorder' in key:
            del st.session_state[key]
    st.rerun()

st.title("📊 Step 1: Native Baseline & Rhythm Analyzer")
st.markdown("원어민의 발화 리듬을 기준으로 여러분의 **연결성**을 분석합니다.")

target_sentence = "The quick brown fox jumps over the lazy dog."
st.info(f"**Read this:** {target_sentence}")

# --- [핵심] 원어민 기준점(Baseline) 생성 ---
@st.cache_resource
def get_native_baseline(text):
    tts = gTTS(text=text, lang='en', tld='com')
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    audio_segment = AudioSegment.from_file(mp3_fp, format="mp3")
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)
    y, sr = librosa.load(wav_io, sr=None)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    duration = librosa.get_duration(y=y_trimmed, sr=sr)
    
    # 에너지 분석
    rms = librosa.feature.rms(y=y_trimmed)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    frame_dur = duration / len(rms)
    is_silent = rms_db < -25
    
    pauses = []
    curr = 0
    for s in is_silent:
        if s: curr += frame_dur
        else:
            if curr > 0: pauses.append(curr)
            curr = 0
            
    return {
        'rate': (len(text.split()) * 1.3 / duration) * 60,
        'staccato': len([d for d in pauses if 0.05 <= d < 0.25]),
        'pause': len([d for d in pauses if d >= 0.25]),
        'audio': y_trimmed, 'sr': sr
    }

native = get_native_baseline(target_sentence)

# --- 2. 녹음 섹션 ---
col_rec, col_reset = st.columns([1, 5])
with col_rec:
    # key를 유동적으로 관리하여 리셋 시 위젯 자체가 초기화되게 함
    audio_data = mic_recorder(
        start_prompt="🔴 녹음 시작",
        stop_prompt="⏹️ 중지 및 분석",
        key='rhythm_recorder_final'
    )

with col_reset:
    if st.button("🔄 다시 시도 (Reset)"):
        reset_app()

# --- 3. 분석 프로세스 ---
if audio_data:
    # 데이터가 들어온 직후 유효성 검사 (아무 소리도 없는 경우 차단)
    audio_bytes = audio_data['bytes']
    if len(audio_bytes) < 3000: # 너무 짧은 파일 차단
        st.warning("녹음이 너무 짧습니다. 문장을 끝까지 읽어주세요.")
        st.stop()

    try:
        with st.spinner("리듬 분석 중..."):
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)
            y, sr = librosa.load(wav_io, sr=None)
            
            # 음량이 너무 작으면 차단
            if np.max(np.abs(y)) < 0.02:
                st.error("음성이 감지되지 않았습니다. 마이크를 확인하고 다시 녹음해주세요.")
                st.stop()

            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
            duration = librosa.get_duration(y=y_trimmed, sr=sr)
            
            # 리듬 분석
            rms = librosa.feature.rms(y=y_trimmed)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            frame_dur = duration / len(rms)
            is_silent = rms_db < -25
            
            pauses = []
            curr = 0
            for s in is_silent:
                if s: curr += frame_dur
                else:
                    if curr > 0: pauses.append(curr)
                    curr = 0
            
            # 결과 저장
            st.session_state.analysis_result = {
                'rate': (len(target_sentence.split()) * 1.3 / duration) * 60,
                'staccato': len([d for d in pauses if 0.05 <= d < 0.25]),
                'pause': len([d for d in pauses if d >= 0.25]),
                'duration': duration
            }
    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")

# --- 4. 결과 출력 (결과가 있을 때만 렌더링) ---
if st.session_state.analysis_result:
    res = st.session_state.analysis_result
    st.divider()
    st.subheader("🎯 원어민 대조 분석 결과")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("발화 속도", f"{res['rate']:.1f} SPM", 
              delta=f"{res['rate'] - native['rate']:.1f} (Target: {native['rate']:.1f})")
    m2.metric("음절 간 끊김", f"{res['staccato']} 회", 
              delta=f"{res['staccato'] - native['staccato']} 회", delta_color="inverse")
    m3.metric("긴 멈춤", f"{res['pause']} 회", 
              delta=f"{res['pause'] - native['pause']} 회", delta_color="inverse")

    # 피드백
    st.write("---")
    st.subheader("💡 Expert Feedback")
    if res['staccato'] >= (len(target_sentence.split()) * 0.4):
        st.error("🚩 **음절마다 끊어 읽기가 많아 보입니다.**")
        st.write("단어들을 하나씩 떼어서 읽기보다는, 소리를 다음 단어까지 부드럽게 밀어내는 **연음(Linking)** 연습을 해보세요.")
    elif res['rate'] < (native['rate'] - 30):
        st.warning("원어민보다 속도가 상당히 느립니다. 좀 더 유창하게 이어 읽어보세요.")
    else:
        st.success("자연스러운 연결성입니다. 훌륭합니다!")
else:
    # 분석 결과가 없을 때는 안내 메시지만 표시
    st.write("---")
    st.info("녹음 버튼을 눌러 발화를 시작하세요. 결과가 나오지 않는다면 다시 녹음해 주세요.")
