import streamlit as st
import librosa
import numpy as np
import io
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
import plotly.graph_objects as go

# --- 1. 상태 초기화 및 페이지 설정 ---
st.set_page_config(page_title="Rhythm Analyzer v2.3", layout="wide")

# 분석 결과 세션 관리
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'widget_id' not in st.session_state:
    st.session_state.widget_id = 0

def reset_app():
    # 모든 결과 지우고 위젯 ID 변경하여 강제 리프레시
    st.session_state.analysis_result = None
    st.session_state.widget_id += 1
    st.rerun()

st.title("📊 Step 1: Rhythm Analyzer (v2.3)")
target_sentence = "The quick brown fox jumps over the lazy dog."
st.info(f"**Read this:** {target_sentence}")

# 원어민 베이스라인 생성 (gTTS)
@st.cache_resource
def get_native_baseline(text):
    tts = gTTS(text=text, lang='en', tld='com')
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    audio = AudioSegment.from_file(mp3_fp, format="mp3")
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    y, sr = librosa.load(wav_io, sr=None)
    y_trim, _ = librosa.effects.trim(y, top_db=25)
    dur = librosa.get_duration(y=y_trim, sr=sr)
    rms = librosa.feature.rms(y=y_trim)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    is_silent = rms_db < -25
    pauses = []
    curr = 0
    for s in is_silent:
        if s: curr += (dur/len(rms))
        else:
            if curr > 0: pauses.append(curr)
            curr = 0
    return {
        'rate': (len(text.split()) * 1.3 / dur) * 60,
        'staccato': len([d for d in pauses if 0.05 <= d < 0.25]),
        'pause': len([d for d in pauses if d >= 0.25])
    }

native = get_native_baseline(target_sentence)

# --- 2. 컨트롤 섹션 ---
col_rec, col_reset = st.columns([1, 5])
with col_rec:
    # widget_id를 key에 포함하여 리셋 시 위젯 자체가 새로고침되도록 함
    audio_data = mic_recorder(
        start_prompt="🔴 녹음 시작",
        stop_prompt="⏹️ 중지 및 분석",
        key=f"recorder_{st.session_state.widget_id}"
    )

with col_reset:
    if st.button("🔄 다시 시도 (Reset)"):
        reset_app()

# --- 3. [핵심] 강화된 노이즈 필터링 및 분석 ---
if audio_data:
    try:
        audio_bytes = audio_data['bytes']
        
        # [검증 1] 파일 용량 체크 (0.1초 미만 녹음 방지)
        if len(audio_bytes) < 3000:
            st.session_state.analysis_result = None
            st.warning("녹음이 너무 짧습니다. 문장을 끝까지 읽어주세요.")
            st.stop()

        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        y, sr = librosa.load(wav_io, sr=None)

        # [검증 2] 음성 에너지 체크 (실제 목소리 여부)
        # 0.05는 일반적인 마이크 잡음보다 큰 수치입니다.
        peak_amp = np.max(np.abs(y))
        if peak_amp < 0.05:
            st.session_state.analysis_result = None
            st.error("⚠️ 음성이 감지되지 않았습니다. 더 큰 목소리로 읽어주세요.")
            st.stop()

        # 분석 진행
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        duration = librosa.get_duration(y=y_trimmed, sr=sr)
        
        # 너무 짧은 오디오 분석 방지
        if duration < 0.5:
            st.session_state.analysis_result = None
            st.warning("유효한 발화 구간이 너무 짧습니다.")
            st.stop()

        # 에너지 기반 휴지(Pause) 분석
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
        
        # 결과값 세션 저장
        st.session_state.analysis_result = {
            'rate': (len(target_sentence.split()) * 1.3 / duration) * 60,
            'staccato': len([d for d in pauses if 0.05 <= d < 0.25]),
            'pause': len([d for d in pauses if d >= 0.25]),
            'peak': peak_amp
        }

    except Exception as e:
        st.session_state.analysis_result = None
        st.error(f"분석 중 오류 발생: {e}")

# --- 4. 결과 렌더링 (결과가 존재할 때만 표시) ---
if st.session_state.analysis_result:
    res = st.session_state.analysis_result
    st.divider()
    st.subheader("🎯 원어민 대조 분석 결과")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("발화 속도", f"{res['rate']:.1f} SPM", 
              delta=f"{res['rate'] - native['rate']:.1f} (Target: {native['rate']:.1f} )")
    m2.metric("음절 간 끊김", f"{res['staccato']} 회", 
              delta=f"{res['staccato'] - native['staccato']} 회", delta_color="inverse")
    m3.metric("긴 멈춤", f"{res['pause']} 회", 
              delta=f"{res['pause'] - native['pause']} 회", delta_color="inverse")

    st.divider()
    st.subheader("💡 Expert Feedback")
    if res['staccato'] >= (len(target_sentence.split()) * 0.4):
        st.error("🚩 **음절마다 끊어 읽기가 많아 보입니다.**")
        st.write("단어들을 하나씩 떼어서 읽기보다는, 소리를 다음 단어까지 부드럽게 밀어내는 **연음(Linking)** 연습을 해보세요.")
    else:
        st.success("자연스러운 연결성입니다. 훌륭합니다!")
else:
    # 결과가 없을 때(초기 상태나 무음 녹음 후)
    st.write("---")
    st.info("녹음 버튼을 누르고 문장을 끝까지 읽어주세요. 무음이나 짧은 소음은 분석되지 않습니다.")
