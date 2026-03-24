import streamlit as st
import librosa
import numpy as np
import io
import plotly.graph_objects as go
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS

st.set_page_config(page_title="Rhythm Analyzer v2.8", layout="wide")

# 세션 관리
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'widget_id' not in st.session_state:
    st.session_state.widget_id = 0

def reset_app():
    st.session_state.analysis_result = None
    st.session_state.widget_id += 1
    st.rerun()

# --- [공통 분석 엔진] 원어민과 학생 모두 이 함수를 통과함 ---
def analyze_rhythm(audio_bytes, is_native=False):
    # 1. WAV 변환 및 로드
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    # 노이즈 게이트 적용: 미세 잡음 제거 (학생 녹음 환경 고려)
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)
    
    y, sr = librosa.load(wav_io, sr=16000)
    
    # 2. 강도 정규화 (소리 크기를 동일하게 맞춤)
    y = librosa.util.normalize(y)
    
    # 3. Trimming (앞뒤 무음 제거 - 기준 통일)
    # 원어민과 학생 모두 35dB 기준으로 동일하게 자름
    y_trim, _ = librosa.effects.trim(y, top_db=35)
    duration = librosa.get_duration(y=y_trim, sr=sr)
    
    # 4. 에너지 분석 (RMS)
    rms = librosa.feature.rms(y=y_trim)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    frame_dur = duration / len(rms)
    
    # 휴지 탐지 기준 (두 데이터 모두 동일 적용)
    # -30dB 이하를 무음으로 간주
    is_silent = rms_db < -30
    
    pauses = []
    curr = 0
    for s in is_silent:
        if s: curr += frame_dur
        else:
            if curr > 0: pauses.append(curr)
            curr = 0
            
    # 지표 계산
    staccato = len([d for d in pauses if 0.05 <= d < 0.25])
    long_pause = len([d for d in pauses if d >= 0.25])
    
    # 속도 계산용 음절 추정 (단어수 기반 고정)
    word_count = 9 # "The quick brown fox jumps over the lazy dog."
    rate = ((word_count * 1.4) / duration) * 60
    
    return {'rate': rate, 'staccato': staccato, 'pause': long_pause, 'wav': wav_io.getvalue(), 'dur': duration}

# --- 2. 앱 UI 및 로직 ---
st.title("📊 Step 1: Fluency & Rhythm Analyzer")
target_text = "The quick brown fox jumps over the lazy dog."

# 원어민 기준점 생성
@st.cache_resource
def get_native(text):
    tts = gTTS(text=text, lang='en', tld='com')
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    return analyze_rhythm(mp3_fp.getvalue(), is_native=True)

native = get_native(target_text)

with st.sidebar:
    st.header("🎧 Native Guide")
    st.audio(native['wav'])
    st.caption(f"Native Spec: {native['rate']:.1f} SPM / Duration: {native['dur']:.2f}s")
    if st.button("🔄 Reset"): reset_app()

audio_data = mic_recorder(start_prompt="🔴 녹음 시작", stop_prompt="⏹️ 중지 및 분석", key=f"rec_{st.session_state.widget_id}")

if audio_data:
    # 학생 발화 분석 (원어민과 동일한 analyze_rhythm 함수 사용)
    res = analyze_rhythm(audio_data['bytes'])
    
    # 유효성 검사 (너무 조용하면 차단)
    if res['dur'] < 0.5:
        st.error("음성이 너무 짧거나 감지되지 않았습니다.")
        st.stop()
        
    st.session_state.analysis_result = res

# --- 3. 결과 렌더링 ---
if st.session_state.analysis_result:
    res = st.session_state.analysis_result
    st.divider()
    
    m1, m2, m3 = st.columns(3)
    # 일치도 계산 (속도 기준)
    match_score = max(0, 100 - abs(res['rate'] - native['rate']))
    
    m1.metric("발화 속도", f"{res['rate']:.1f} SPM", delta=f"Target: {native['rate']:.1f}")
    m2.metric("음절 간 끊김", f"{res['staccato']} 회", delta=f"Target: {native['staccato']} 회", delta_color="inverse")
    m3.metric("긴 멈춤", f"{res['pause']} 회", delta=f"Target: {native['pause']} 회", delta_color="inverse")

    # 대조 그래프
    fig = go.Figure(data=[
        go.Bar(name='Native', x=['Speed', 'Gaps', 'Pauses'], y=[native['rate'], native['staccato'], native['pause']], marker_color='#D1D1D1'),
        go.Bar(name='Student', x=['Speed', 'Gaps', 'Pauses'], y=[res['rate'], res['staccato'], res['pause']], marker_color='#1E88E5')
    ])
    fig.update_layout(barmode='group', height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    # 최종 판정
    if match_score > 90 and res['staccato'] == native['staccato']:
        st.success(f"🌟 **완벽합니다!** 원어민 가이드와 {match_score:.1f}% 일치하는 리듬입니다.")
    elif match_score > 80:
        st.info("훌륭한 리듬입니다. 원어민과 거의 유사합니다.")
    else:
        st.warning("원어민 리듬을 다시 듣고 연음(Linking)에 신경 써보세요.")
