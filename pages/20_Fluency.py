import streamlit as st
import librosa
import numpy as np
import io
import plotly.graph_objects as go
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS

# --- 1. 초기화 및 설정 ---
st.set_page_config(page_title="Rhythm Analyzer v3.1", layout="wide")

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'widget_id' not in st.session_state:
    st.session_state.widget_id = 0

def reset_app():
    st.session_state.analysis_result = None
    st.session_state.widget_id += 1
    st.rerun()

# --- [정밀 분석 엔진] ---
def analyze_rhythm(audio_bytes, is_student=False):
    # A. 기본 로드 및 규격화
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)
    
    y, sr = librosa.load(wav_io, sr=16000)
    
    # [강력 필터 1] 에너지 체크 (노이즈와 실제 음성 구분)
    # 전체 음량의 RMS 평균이 일정 수준 이하면 분석 거부
    rms_mean = np.sqrt(np.mean(y**2))
    if is_student and rms_mean < 0.01: # 0.01 이하는 사실상 주변 소음
        return None

    y = librosa.util.normalize(y)
    
    # [강력 필터 2] 무음 제거 후 길이 체크
    y_trim, _ = librosa.effects.trim(y, top_db=30)
    duration = librosa.get_duration(y=y_trim, sr=sr)
    
    # 문장(9단어)을 읽는데 1.5초 미만은 물리적으로 불가능 (노이즈로 간주)
    if is_student and duration < 1.5: 
        return None
    
    # B. 리듬 분석
    rms = librosa.feature.rms(y=y_trim)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    frame_dur = duration / len(rms)
    
    is_silent = rms_db < -30
    pauses = []
    curr = 0
    for s in is_silent:
        if s: curr += frame_dur
        else:
            if curr > 0: pauses.append(curr)
            curr = 0
            
    word_count = 9 
    rate = ((word_count * 1.4) / duration) * 60
    
    return {
        'rate': rate, 
        'staccato': len([d for d in pauses if 0.05 <= d < 0.25]), 
        'pause': len([d for d in pauses if d >= 0.25]), 
        'wav': wav_io.getvalue(), 
        'dur': duration
    }

# --- 2. 원어민 데이터 (Target) ---
target_text = "The quick brown fox jumps over the lazy dog."

@st.cache_resource
def get_native(text):
    tts = gTTS(text=text, lang='en', tld='com')
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    return analyze_rhythm(mp3_fp.getvalue(), is_student=False)

native = get_native(target_text)

# --- 3. 메인 UI ---
st.title("📊 Step 1: Fluency & Rhythm Analyzer")

with st.container(border=True):
    st.subheader("🎧 원어민 가이드 (Native Guide)")
    st.info(f"**Target:** {target_text}")
    st.audio(native['wav'], format="audio/wav")

st.divider()

st.subheader("🎙️ 나의 발화 연습")
col_rec, col_reset = st.columns([1, 4])

with col_rec:
    audio_data = mic_recorder(
        start_prompt="🔴 녹음 시작",
        stop_prompt="⏹️ 중지 및 분석",
        key=f"rec_{st.session_state.widget_id}"
    )

with col_reset:
    if st.session_state.analysis_result:
        if st.button("🔄 다시 시도 (Reset)"):
            reset_app()

# --- 4. 데이터 검증 및 분석 ---
if audio_data:
    # 매 녹음 시도마다 결과 초기화 후 다시 검증
    res = analyze_rhythm(audio_data['bytes'], is_student=True)
    if res:
        st.session_state.analysis_result = res
    else:
        st.session_state.analysis_result = None # 유효하지 않으면 결과 비움
        st.error("⚠️ 음성이 너무 짧거나 감지되지 않았습니다. 문장 전체를 명확하게 읽어주세요.")

# --- 5. 결과 렌더링 ---
if st.session_state.analysis_result:
    res = st.session_state.analysis_result
    st.write("### 👂 내 목소리 다시 듣기")
    st.audio(res['wav'], format="audio/wav")
    
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("발화 속도", f"{res['rate']:.1f} SPM", delta=f"Target: {native['rate']:.1f}")
    m2.metric("음절 간 끊김", f"{res['staccato']} 회", delta=f"Target: {native['staccato']} 회", delta_color="inverse")
    m3.metric("긴 멈춤", f"{res['pause']} 회", delta=f"Target: {native['pause']} 회", delta_color="inverse")

    fig = go.Figure(data=[
        go.Bar(name='Native', x=['Speed', 'Gaps', 'Pauses'], y=[native['rate'], native['staccato'], native['pause']], marker_color='#D1D1D1'),
        go.Bar(name='You', x=['Speed', 'Gaps', 'Pauses'], y=[res['rate'], res['staccato'], res['pause']], marker_color='#1E88E5')
    ])
    fig.update_layout(barmode='group', height=400)
    st.plotly_chart(fig, use_container_width=True)
