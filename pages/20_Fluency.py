import streamlit as st
import librosa
import numpy as np
import io
import plotly.graph_objects as go
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS

# --- 1. 상태 초기화 및 페이지 설정 ---
st.set_page_config(page_title="Rhythm Analyzer v2.4", layout="wide")

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'widget_id' not in st.session_state:
    st.session_state.widget_id = 0

def reset_app():
    st.session_state.analysis_result = None
    st.session_state.widget_id += 1
    st.rerun()

st.title("📊 Step 1: Fluency & Rhythm Analyzer")
st.markdown("본 서비스는 원어민의 발화 리듬을 **참고 가이드**로 제공합니다.")

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
    audio_data = mic_recorder(
        start_prompt="🔴 녹음 시작",
        stop_prompt="⏹️ 중지 및 분석",
        key=f"recorder_{st.session_state.widget_id}"
    )
with col_reset:
    if st.button("🔄 다시 시도 (Reset)"):
        reset_app()

# --- 3. 분석 프로세스 ---
if audio_data:
    try:
        audio_bytes = audio_data['bytes']
        # 무음/단기 녹음 필터
        if len(audio_bytes) < 5000:
            st.session_state.analysis_result = None
            st.stop()

        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        y, sr = librosa.load(wav_io, sr=None)

        # 에너지 필터 (진폭 0.05 미만 차단)
        if np.max(np.abs(y)) < 0.05:
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
        st.error(f"분석 중 오류 발생: {e}")

# --- 4. 결과 출력 ---
if st.session_state.analysis_result:
    res = st.session_state.analysis_result
    st.divider()
    
    # [수정] 원어민 대조 지표 레이아웃
    st.subheader("🏁 원어민 지표와 비교 (참고용)")
    m1, m2, m3 = st.columns(3)
    m1.metric("나의 발화 속도", f"{res['rate']:.1f} SPM", delta=f"목표: {native['rate']:.1f}")
    m2.metric("음절 간 끊김", f"{res['staccato']} 회", delta=f"원어민: {native['staccato']} 회", delta_color="inverse")
    m3.metric("긴 멈춤", f"{res['pause']} 회", delta=f"원어민: {native['pause']} 회", delta_color="inverse")

    # [추가] 시각화 도표 복구
    st.write("")
    fig = go.Figure(data=[
        go.Bar(name='원어민 (Guide)', x=['발화 속도', '음절 간 끊김', '긴 멈춤'], 
               y=[native['rate'], native['staccato'], native['pause']], marker_color='#E0E0E0'),
        go.Bar(name='나 (Student)', x=['발화 속도', '음절 간 끊김', '긴 멈춤'], 
               y=[res['rate'], res['staccato'], res['pause']], marker_color='#2E7D32')
    ])
    fig.update_layout(barmode='group', height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # [수정] 교육적 관점의 피드백
    st.divider()
    st.subheader("💡 학습 가이드 피드백")
    
    # 탭을 사용하여 원어민과의 차이점을 부드럽게 설명
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if res['staccato'] > native['staccato'] + 2:
            st.error("💡 개선 포인트")
        else:
            st.success("🌟 잘하고 있어요!")
            
    with col_b:
        if res['staccato'] >= (len(target_sentence.split()) * 0.4):
            st.write("**음절마다 끊어 읽기가 원어민보다 많은 편입니다.**")
            st.write("원어민의 리듬은 단어들이 하나의 선처럼 연결됩니다. 개별 단어의 발음보다 앞 단어와 뒷 단어를 부드럽게 잇는 **연음(Linking)**에 집중해서 다시 시도해 보세요.")
        elif res['rate'] < (native['rate'] - 40):
            st.write("**원어민의 속도를 참고하여 조금 더 자신감 있게 읽어보세요.**")
            st.write("속도가 너무 느리면 문장의 의미 단위(Chunk)가 깨질 수 있습니다. 원어민의 속도는 절대적인 기준은 아니지만, 흐름을 타는 데 도움이 됩니다.")
        else:
            st.write("**원어민의 리듬과 매우 흡사한 흐름을 보여주고 있습니다.**")
            st.write("지금처럼 의미 단위로 묶어서 읽는 습관을 유지하세요. 매우 자연스러운 발화입니다.")
else:
    st.write("---")
    st.info("좌측 상단의 버튼을 눌러 녹음을 시작하세요. 원어민의 수치는 여러분의 개선 방향을 돕는 가이드라인입니다.")
