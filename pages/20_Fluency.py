import streamlit as st
import librosa
import numpy as np
import io
import plotly.graph_objects as go
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS

# --- 1. 상태 초기화 및 페이지 설정 ---
st.set_page_config(page_title="Rhythm Analyzer v2.5", layout="wide")

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'widget_id' not in st.session_state:
    st.session_state.widget_id = 0

def reset_app():
    st.session_state.analysis_result = None
    st.session_state.widget_id += 1
    st.rerun()

st.title("📊 Step 1: Fluency & Rhythm Analyzer")
st.markdown("원어민의 발화 리듬을 **참고 가이드**로 활용하여 자신의 연결성을 점검해보세요.")

target_sentence = "The quick brown fox jumps over the lazy dog."
st.info(f"**Target Sentence:** {target_sentence}")

# --- [복구] 원어민 베이스라인 생성 및 음성 추출 ---
@st.cache_resource
def get_native_baseline(text):
    tts = gTTS(text=text, lang='en', tld='com')
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    
    # 분석용 데이터 생성
    audio = AudioSegment.from_file(mp3_fp, format="mp3")
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    
    y, sr = librosa.load(wav_io, sr=None)
    y_trim, _ = librosa.effects.trim(y, top_db=25)
    dur = librosa.get_duration(y=y_trim, sr=sr)
    
    # 에너지 분석 로직
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
            
    # 재생용 오디오 바이트 보관
    return {
        'rate': (len(text.split()) * 1.3 / dur) * 60,
        'staccato': len([d for d in pauses if 0.05 <= d < 0.25]),
        'pause': len([d for d in pauses if d >= 0.25]),
        'wav_bytes': wav_io.getvalue() # 재생용
    }

native = get_native_baseline(target_sentence)

# --- 사이드바: 원어민 가이드 리스닝 ---
with st.sidebar:
    st.header("🎧 Native Guide")
    st.write("분석 전 원어민의 리듬을 들어보세요.")
    st.audio(native['wav_bytes'], format="audio/wav")
    st.divider()
    if st.button("🔄 전체 초기화 (Reset All)"):
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

# --- 3. 분석 프로세스 (검증 강화) ---
if audio_data:
    try:
        audio_bytes = audio_data['bytes']
        if len(audio_bytes) < 5000: # 최소 용량 미달 시 무시
            st.session_state.analysis_result = None
            st.stop()

        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        y, sr = librosa.load(wav_io, sr=None)

        if np.max(np.abs(y)) < 0.05: # 무음 필터
            st.session_state.analysis_result = None
            st.stop()

        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        duration = librosa.get_duration(y=y_trimmed, sr=sr)
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
        
        st.session_state.analysis_result = {
            'rate': (len(target_sentence.split()) * 1.3 / duration) * 60,
            'staccato': len([d for d in pauses if 0.05 <= d < 0.25]),
            'pause': len([d for d in pauses if d >= 0.25])
        }
    except Exception as e:
        st.error(f"분석 오류: {e}")

# --- 4. 결과 출력 ---
if st.session_state.analysis_result:
    res = st.session_state.analysis_result
    st.divider()
    
    st.subheader("🎯 원어민 지표 대조 (참고 가이드)")
    m1, m2, m3 = st.columns(3)
    m1.metric("나의 속도", f"{res['rate']:.1f} SPM", delta=f"원어민: {native['rate']:.1f}")
    m2.metric("음절 간 끊김", f"{res['staccato']} 회", delta=f"원어민: {native['staccato']} 회", delta_color="inverse")
    m3.metric("긴 멈춤", f"{res['pause']} 회", delta=f"원어민: {native['pause']} 회", delta_color="inverse")

    # 시각화 도표
    st.write("")
    fig = go.Figure(data=[
        go.Bar(name='원어민 (Native)', x=['발화 속도', '음절 간 끊김', '긴 멈춤'], 
               y=[native['rate'], native['staccato'], native['pause']], marker_color='#D1D1D1'),
        go.Bar(name='나 (Student)', x=['발화 속도', '음절 간 끊김', '긴 멈춤'], 
               y=[res['rate'], res['staccato'], res['pause']], marker_color='#1E88E5')
    ])
    fig.update_layout(barmode='group', height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 교육적 피드백
    st.divider()
    st.subheader("💡 Expert Feedback")
    
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if res['staccato'] > native['staccato'] + 2:
            st.error("💡 Focus Area")
        else:
            st.success("🌟 Good Flow!")
            
    with col_b:
        if res['staccato'] >= (len(target_sentence.split()) * 0.4):
            st.write("**음절마다 끊어 읽기가 원어민보다 많은 편입니다.**")
            st.write("개별 단어의 발음도 중요하지만, 단어와 단어 사이를 멈추지 않고 소리를 밀어내는 **연음(Linking)** 연습이 더 필요해 보입니다. 원어민의 음성을 다시 들으며 리듬을 따라해보세요.")
        elif res['rate'] < (native['rate'] - 40):
            st.write("**전반적인 속도를 조금 더 높여볼까요?**")
            st.write("속도가 너무 느리면 문장의 의미가 분절되어 들릴 수 있습니다. 원어민의 속도감을 참고하여 조금 더 유창하게 읽는 연습을 추천합니다.")
        else:
            st.write("**훌륭합니다! 원어민의 리듬과 매우 흡사하게 발화하셨습니다.**")
            st.write("의미 단위별로 호흡을 조절하는 능력이 뛰어납니다. 이 흐름을 유지하세요.")
else:
    st.write("---")
    st.info("좌측 상단의 버튼을 눌러 녹음을 시작하세요. 사이드바에서 원어민의 발음을 먼저 들어볼 수 있습니다.")
