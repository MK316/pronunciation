import streamlit as st
import librosa
import numpy as np
import io
import plotly.graph_objects as go
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS

st.set_page_config(page_title="Rhythm Analyzer v3.3", layout="wide")

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'widget_id' not in st.session_state:
    st.session_state.widget_id = 0

def reset_app():
    st.session_state.analysis_result = None
    st.session_state.widget_id += 1
    st.rerun()

# --- [분석 및 점수 산출 엔진] ---
def analyze_and_score(audio_bytes, is_student=False):
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)
    y, sr = librosa.load(wav_io, sr=16000)
    
    # 노이즈 필터
    rms_mean = np.sqrt(np.mean(y**2))
    if is_student and rms_mean < 0.01: return None

    y = librosa.util.normalize(y)
    y_trim, _ = librosa.effects.trim(y, top_db=30)
    duration = librosa.get_duration(y=y_trim, sr=sr)
    
    if is_student and duration < 1.5: return None
    
    # 리듬 분석
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
            
    staccato = len([d for d in pauses if 0.05 <= d < 0.25])
    long_pause = len([d for d in pauses if d >= 0.25])
    
    # SPS (Syllables Per Second) 계산
    word_count = 9
    syllables = word_count * 1.45
    sps = syllables / duration
    rate_spm = (syllables / duration) * 60
    
    # --- [점수 산출 로직] ---
    # 1. 속도 점수 (60점 만점): 3.5~5.5 SPS 기준
    if 3.5 <= sps <= 5.5:
        speed_score = 60
    else:
        # 범위를 벗어날수록 감점 (최소 20점 보장)
        dist = min(abs(sps - 3.5), abs(sps - 5.5))
        speed_score = max(20, 60 - (dist * 20))
        
    # 2. 연결성 점수 (40점 만점)
    conn_score = max(0, 40 - (staccato * 5) - (long_pause * 10))
    
    total_score = int(speed_score + conn_score)
    
    return {
        'rate': rate_spm, 'sps': sps, 'staccato': staccato, 'pause': long_pause,
        'score': total_score, 'wav': wav_io.getvalue(), 'dur': duration
    }

# --- 2. 원어민 기준 데이터 ---
target_text = "The quick brown fox jumps over the lazy dog."
@st.cache_resource
def get_native(text):
    tts = gTTS(text=text, lang='en', tld='com')
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    return analyze_and_score(mp3_fp.getvalue(), is_student=False)

native = get_native(target_text)

# --- 3. 메인 UI ---
st.title("📊 Step 1: Fluency Score Analyzer")
st.info(f"**Target:** {target_text} (적정 속도: 3.5~5.5 SPS)")

with st.expander("🎧 원어민 가이드 들어보기", expanded=True):
    st.audio(native['wav'], format="audio/wav")

st.divider()

col_rec, col_reset = st.columns([1, 4])
with col_rec:
    audio_data = mic_recorder(start_prompt="🔴 녹음 시작", stop_prompt="⏹️ 중지 및 분석", key=f"rec_{st.session_state.widget_id}")
with col_reset:
    if st.session_state.analysis_result:
        if st.button("🔄 다시 시도 (Reset)"): reset_app()

if audio_data:
    res = analyze_and_score(audio_data['bytes'], is_student=True)
    if res: st.session_state.analysis_result = res
    else: st.error("⚠️ 음성이 감지되지 않았습니다. 다시 녹음해주세요.")

# --- 4. 결과 및 점수 출력 ---
if st.session_state.analysis_result:
    res = st.session_state.analysis_result
    
    # 점수에 따른 색상 지정
    score_color = "green" if res['score'] >= 80 else "orange" if res['score'] >= 60 else "red"
    
    st.markdown(f"### 🏆 유창성 점수: :{score_color}[{res['score']}점] / 100점")
    st.progress(res['score'] / 100)
    
    st.audio(res['wav'], format="audio/wav")
    
    # 메트릭 및 이중 축 그래프 (이전 로직 유지)
    m1, m2, m3 = st.columns(3)
    m1.metric("초당 음절 수 (SPS)", f"{res['sps']:.1f}", delta=f"Target: 3.5~5.5")
    m2.metric("음절 간 끊김", f"{res['staccato']} 회", delta_color="inverse")
    m3.metric("긴 멈춤", f"{res['pause']} 회", delta_color="inverse")

    # 이중 축 그래프 생략 (코드 간소화, 기존 Plotly 로직 그대로 사용 가능)
    
    # 학술적 피드백
    st.divider()
    st.subheader("💡 Expert Feedback")
    if res['score'] >= 90:
        st.success("원어민 수준의 완벽한 유창성입니다! 속도와 연결성 모두 훌륭합니다.")
    elif 3.5 <= res['sps'] <= 5.5:
        if res['staccato'] > 2:
            st.warning("속도는 적절하지만 음절 간 끊김(Staccato)이 잦습니다. 연음(Linking)에 더 집중해보세요.")
        else:
            st.info("안정적인 속도입니다. 멈춤 구간을 조금 더 줄이면 만점에 가까워질 수 있습니다.")
    else:
        st.error("발화 속도가 최적 범위(3.5~5.5 SPS)를 벗어났습니다. 너무 빠르거나 느리지 않게 조절하는 연습이 필요합니다.")
