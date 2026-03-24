import streamlit as st
import librosa
import numpy as np
import io
import plotly.graph_objects as go
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS

st.set_page_config(page_title="Rhythm Analyzer v2.6", layout="wide")

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'widget_id' not in st.session_state:
    st.session_state.widget_id = 0

def reset_app():
    st.session_state.analysis_result = None
    st.session_state.widget_id += 1
    st.rerun()

st.title("📊 Step 1: Fluency & Rhythm Analyzer")
target_sentence = "The quick brown fox jumps over the lazy dog."
st.info(f"**Target Sentence:** {target_sentence}")

# --- [수정] 원어민 베이스라인 생성 로직 보정 ---
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
    # [보정] 너무 공격적인 trim 방지 (top_db를 30으로 조정)
    y_trim, _ = librosa.effects.trim(y, top_db=30)
    dur = librosa.get_duration(y=y_trim, sr=sr)
    
    # [보정] gTTS 특유의 빠른 속도를 고려한 음절 가중치 조정
    word_count = len(text.split())
    syllable_est = word_count * 1.45 
    
    # 멈춤 분석
    rms = librosa.feature.rms(y=y_trim)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    frame_dur = dur / len(rms)
    is_silent = rms_db < -30 # 기준 완화
    
    pauses = []
    curr = 0
    for s in is_silent:
        if s: curr += frame_dur
        else:
            if curr > 0: pauses.append(curr)
            curr = 0
            
    return {
        'rate': (syllable_est / dur) * 60,
        'staccato': len([d for d in pauses if 0.06 <= d < 0.25]),
        'pause': len([d for d in pauses if d >= 0.25]),
        'wav_bytes': wav_io.getvalue(),
        'duration': dur
    }

native = get_native_baseline(target_sentence)

# --- 사이드바 ---
with st.sidebar:
    st.header("🎧 Native Guide")
    st.audio(native['wav_bytes'], format="audio/wav")
    st.caption(f"Native Duration: {native['duration']:.2f}s")
    st.caption(f"Native Speed: {native['rate']:.1f} SPM")
    if st.button("🔄 전체 초기화 (Reset)"):
        reset_app()

# --- 2. 컨트롤 섹션 ---
col_rec, col_reset = st.columns([1, 5])
with col_rec:
    audio_data = mic_recorder(
        start_prompt="🔴 녹음 시작",
        stop_prompt="⏹️ 중지 및 분석",
        key=f"recorder_{st.session_state.widget_id}"
    )
with col_reset:
    if st.session_state.analysis_result:
        if st.button("🔄 다시 시도 (Try Again)"):
            reset_app()

# --- 3. 분석 프로세스 ---
if audio_data:
    try:
        audio_bytes = audio_data['bytes']
        if len(audio_bytes) < 5000: st.stop()

        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        y, sr = librosa.load(wav_io, sr=None)

        if np.max(np.abs(y)) < 0.05: st.stop()

        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        duration = librosa.get_duration(y=y_trimmed, sr=sr)
        
        rms = librosa.feature.rms(y=y_trimmed)[0]
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
        
        word_count = len(target_sentence.split())
        st.session_state.analysis_result = {
            'rate': ((word_count * 1.45) / duration) * 60,
            'staccato': len([d for d in pauses if 0.06 <= d < 0.25]),
            'pause': len([d for d in pauses if d >= 0.25]),
            'duration': duration
        }
    except Exception as e:
        st.error(f"분석 오류: {e}")

# --- 4. 결과 출력 ---
if st.session_state.analysis_result:
    res = st.session_state.analysis_result
    st.divider()
    
    # 지표 시각화
    m1, m2, m3 = st.columns(3)
    m1.metric("나의 속도", f"{res['rate']:.1f} SPM", delta=f"Target: {native['rate']:.1f}")
    m2.metric("음절 간 끊김", f"{res['staccato']} 회", delta=f"Native: {native['staccato']} 회", delta_color="inverse")
    m3.metric("긴 멈춤", f"{res['pause']} 회", delta=f"Native: {native['pause']} 회", delta_color="inverse")

    # 대조 그래프
    fig = go.Figure(data=[
        go.Bar(name='Native (Guide)', x=['Speed', 'Gaps', 'Pauses'], 
               y=[native['rate'], native['staccato'], native['pause']], marker_color='#D1D1D1'),
        go.Bar(name='You (Student)', x=['Speed', 'Gaps', 'Pauses'], 
               y=[res['rate'], res['staccato'], res['pause']], marker_color='#1E88E5')
    ])
    fig.update_layout(barmode='group', height=350)
    st.plotly_chart(fig, use_container_width=True)

    # 피드백
    st.divider()
    if abs(res['rate'] - native['rate']) < 15 and res['staccato'] <= native['staccato'] + 1:
        st.success("🌟 **놀랍습니다! 원어민 가이드와 거의 동일한 리듬입니다.**")
        st.write("의미 단위별 연결성과 속도가 매우 이상적입니다.")
    elif res['staccato'] > native['staccato'] + 2:
        st.error("🚩 **음절마다 끊어 읽기가 많아 보입니다.**")
        st.write("단어 사이를 멈추지 말고 소리를 연결하는 연음(Linking) 연습을 해보세요.")
    else:
        st.info("원어민의 속도와 리듬을 참고하여 조금 더 유창하게 다듬어보세요.")
