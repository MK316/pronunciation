import streamlit as st
import librosa
import numpy as np
import io
import plotly.graph_objects as go
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS

# --- 1. 페이지 설정 및 초기화 ---
st.set_page_config(page_title="Step 1: Rhythm Analyzer", layout="wide")

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'widget_id' not in st.session_state:
    st.session_state.widget_id = 0

def reset_app():
    st.session_state.analysis_result = None
    st.session_state.widget_id += 1
    st.rerun()

# --- [공통 분석 엔진] ---
def analyze_rhythm(audio_bytes):
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)
    
    y, sr = librosa.load(wav_io, sr=16000)
    y = librosa.util.normalize(y)
    
    # 공통 Trimming 기준 (35dB)
    y_trim, _ = librosa.effects.trim(y, top_db=35)
    duration = librosa.get_duration(y=y_trim, sr=sr)
    
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

# --- 2. 원어민 기준점 데이터 생성 ---
target_text = "The quick brown fox jumps over the lazy dog."

@st.cache_resource
def get_native(text):
    tts = gTTS(text=text, lang='en', tld='com')
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    return analyze_rhythm(mp3_fp.getvalue())

native = get_native(target_text)

# --- 3. 메인 UI 구성 ---
st.title("📊 Step 1: Fluency & Rhythm Analyzer")
st.write("원어민의 리듬과 자신의 발화를 직접 비교해보세요.")

# 가이드 섹션 (메인 창)
with st.container(border=True):
    st.subheader("🎧 Native Guide Audio")
    st.info(f"**Target:** {target_text}")
    st.audio(native['wav'], format="audio/wav")

st.divider()

# 녹음 섹션
st.subheader("🎙️ Your Practice")
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

# --- 4. 분석 및 결과 출력 ---
if audio_data:
    res = analyze_rhythm(audio_data['bytes'])
    if res['dur'] < 0.5:
        st.warning("발화가 너무 짧습니다. 다시 녹음해주세요.")
    else:
        st.session_state.analysis_result = res

if st.session_state.analysis_result:
    res = st.session_state.analysis_result
    
    # [추가] 본인 녹음 다시 듣기
    st.write("### 👂 Listen to Your Recording")
    st.audio(res['wav'], format="audio/wav")
    
    st.divider()
    st.subheader("🎯 Analysis Comparison")
    
    # 메트릭 표시
    m1, m2, m3 = st.columns(3)
    m1.metric("발화 속도", f"{res['rate']:.1f} SPM", delta=f"Target: {native['rate']:.1f}")
    m2.metric("음절 간 끊김", f"{res['staccato']} 회", delta=f"Target: {native['staccato']} 회", delta_color="inverse")
    m3.metric("긴 멈춤", f"{res['pause']} 회", delta=f"Target: {native['pause']} 회", delta_color="inverse")

    # 대조 그래프
    fig = go.Figure(data=[
        go.Bar(name='Native', x=['Speed', 'Syllable Gaps', 'Long Pauses'], 
               y=[native['rate'], native['staccato'], native['pause']], marker_color='#D1D1D1'),
        go.Bar(name='You', x=['Speed', 'Syllable Gaps', 'Long Pauses'], 
               y=[res['rate'], res['staccato'], res['pause']], marker_color='#1E88E5')
    ])
    fig.update_layout(barmode='group', height=400)
    st.plotly_chart(fig, use_container_width=True)

    # 피드백
    st.info("💡 **Tip:** 원어민 오디오와 자신의 녹음을 번갈아 들으며 어느 구간에서 리듬 차이가 발생하는지 직접 비교해보세요.")
