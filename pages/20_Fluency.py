import streamlit as st
import librosa
import numpy as np
import io
import plotly.graph_objects as go
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS

# --- 1. 초기화 및 설정 ---
st.set_page_config(page_title="Rhythm Analyzer v3.2", layout="wide")

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
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)
    
    y, sr = librosa.load(wav_io, sr=16000)
    
    # [강력 필터 1] 에너지 체크 (노이즈 게이트)
    rms_mean = np.sqrt(np.mean(y**2))
    if is_student and rms_mean < 0.01: 
        return None

    y = librosa.util.normalize(y)
    
    # [강력 필터 2] 길이 체크 (트리밍 후)
    y_trim, _ = librosa.effects.trim(y, top_db=30)
    duration = librosa.get_duration(y=y_trim, sr=sr)
    
    # 노이즈로 간주 (1.5초 미만 발화)
    if is_student and duration < 1.5: 
        return None
    
    # 에너지 기반 리듬 분석
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
    # 원어민 데이터는 필터 제외
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
    st.subheader("🎯 분석 결과 대조 (이중 축 적용)")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("발화 속도", f"{res['rate']:.1f} SPM", delta=f"Target: {native['rate']:.1f}")
    m2.metric("음절 간 끊김", f"{res['staccato']} 회", delta=f"Target: {native['staccato']} 회", delta_color="inverse")
    m3.metric("긴 멈춤", f"{res['pause']} 회", delta=f"Target: {native['pause']} 회", delta_color="inverse")

    # --- [핵심 수정] 이중 Y축 기반 시각화 ---
    fig = go.Figure()

    # 왼쪽 Y축 (y1): 발화 속도 (SPM) 전용
    fig.add_trace(go.Bar(
        name='Native (Speed)',
        x=['Speed'], 
        y=[native['rate']], 
        marker_color='#D1D1D1', 
        yaxis='y1'
    ))
    fig.add_trace(go.Bar(
        name='You (Speed)',
        x=['Speed'], 
        y=[res['rate']], 
        marker_color='#1E88E5', 
        yaxis='y1'
    ))

    # 오른쪽 Y축 (y2): 멈춤 횟수 (Counts) 전용
    # Gaps와 Pauses를 나란히 배치하기 위해 Grouped Bar 설정
    x_counts = ['Gaps (Staccato)', 'Pauses (Long)']
    fig.add_trace(go.Bar(
        name='Native (Counts)',
        x=x_counts, 
        y=[native['staccato'], native['pause']], 
        marker_color='#9E9E9E', # 조금 더 진한 회색 
        yaxis='y2'
    ))
    fig.add_trace(go.Bar(
        name='You (Counts)',
        x=x_counts, 
        y=[res['staccato'], res['pause']], 
        marker_color='#D32F2F', # 빨간색으로 시인성 확보
        yaxis='y2'
    ))

    # 레이아웃 설정: 이중 축 정의
    fig.update_layout(
        height=450,
        barmode='group', # 막대를 카테고리별로 그룹화
        xaxis=dict(title_text='Rhythm Metrics'),
        yaxis=dict(
            title=dict(text='Speech Rate (SPM)', font=dict(color='#1E88E5')),
            tickfont=dict(color='#1E88E5')
        ),
        yaxis2=dict(
            title=dict(text='Pause Counts', font=dict(color='#D32F2F')),
            tickfont=dict(color='#D32F2F'),
            overlaying='y', # y축 위에 겹쳐서 표시
            side='right' # 오른쪽에 배치
        ),
        legend=dict(x=1.1, y=1) # 범례를 약간 밖으로 이동
    )
    
    st.plotly_chart(fig, use_container_width=True)
