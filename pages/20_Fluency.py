import streamlit as st
import librosa
import numpy as np
import io
import plotly.graph_objects as go
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS

st.set_page_config(page_title="Rhythm Analyzer v2.7", layout="wide")

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

# --- [수정] 원어민 베이스라인 생성 로직 (속도 왜곡 방지) ---
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
    # [핵심 수정] 원어민 음원은 무음 제거를 아주 살짝만 함 (top_db를 60으로 완화)
    y_trim, _ = librosa.effects.trim(y, top_db=60)
    dur = librosa.get_duration(y=y_trim, sr=sr)
    
    # 음절 수 계산 (단어 수 * 가중치 1.45)
    word_count = len(text.split())
    syllables = word_count * 1.45
    
    # 에너지 기반 휴지 분석 (원어민은 거의 0이어야 함)
    rms = librosa.feature.rms(y=y_trim)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    is_silent = rms_db < -40 # 아주 미세한 소리도 발화로 인정
    
    pauses = []
    curr = 0
    frame_dur = dur / len(rms)
    for s in is_silent:
        if s: curr += frame_dur
        else:
            if curr > 0: pauses.append(curr)
            curr = 0
            
    return {
        'rate': (syllables / dur) * 60,
        'staccato': len([d for d in pauses if 0.05 <= d < 0.25]),
        'pause': len([d for d in pauses if d >= 0.25]),
        'wav_bytes': wav_io.getvalue(),
        'duration': dur
    }

native = get_native_baseline(target_sentence)

# --- 사이드바 ---
with st.sidebar:
    st.header("🎧 Native Guide")
    st.audio(native['wav_bytes'], format="audio/wav")
    st.caption(f"Native Speed: {native['rate']:.1f} SPM") # 이제 130~150 사이가 나올 것임
    if st.button("🔄 전체 초기화 (Reset All)"):
        reset_app()

# --- 2. 녹음 섹션 ---
col_rec, _ = st.columns([1, 5])
with col_rec:
    audio_data = mic_recorder(
        start_prompt="🔴 녹음 시작",
        stop_prompt="⏹️ 중지 및 분석",
        key=f"recorder_{st.session_state.widget_id}"
    )

# --- 3. 분석 프로세스 ---
if audio_data:
    try:
        audio_bytes = audio_data['bytes']
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        y, sr = librosa.load(wav_io, sr=None)

        if np.max(np.abs(y)) < 0.05: st.stop()

        # 학생 발화는 노이즈가 있을 수 있으므로 top_db=30 적용
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        duration = librosa.get_duration(y=y_trimmed, sr=sr)
        
        rms = librosa.feature.rms(y=y_trimmed)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        frame_dur = duration / len(rms)
        is_silent = rms_db < -30 # 학생용 기준
        
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
            'staccato': len([d for d in pauses if 0.05 <= d < 0.25]),
            'pause': len([d for d in pauses if d >= 0.25])
        }
    except Exception as e:
        st.error(f"분석 오류: {e}")

# --- 4. 결과 출력 (도표 및 피드백) ---
if st.session_state.analysis_result:
    res = st.session_state.analysis_result
    st.divider()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("나의 속도", f"{res['rate']:.1f} SPM", delta=f"Guide: {native['rate']:.1f}")
    m2.metric("음절 간 끊김", f"{res['staccato']} 회", delta=f"Native: {native['staccato']} 회", delta_color="inverse")
    m3.metric("긴 멈춤", f"{res['pause']} 회", delta=f"Native: {native['pause']} 회", delta_color="inverse")

    # [중요] 도표 복구
    fig = go.Figure(data=[
        go.Bar(name='Native (Guide)', x=['Speed', 'Gaps', 'Pauses'], 
               y=[native['rate'], native['staccato'], native['pause']], marker_color='#D1D1D1'),
        go.Bar(name='You (Student)', x=['Speed', 'Gaps', 'Pauses'], 
               y=[res['rate'], res['staccato'], res['pause']], marker_color='#1E88E5')
    ])
    fig.update_layout(barmode='group', height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    # 속도 차이가 20 SPM 이내면 동일한 것으로 간주
    if abs(res['rate'] - native['rate']) < 20 and res['staccato'] <= native['staccato'] + 1:
        st.success("🌟 **훌륭합니다! 원어민 가이드와 일치하는 리듬을 보여주고 있습니다.**")
    else:
        st.info("원어민 리듬을 참고하여 연음(Linking)과 속도를 조절해 보세요.")
    
    if st.button("🔄 Try Again"):
        reset_app()
