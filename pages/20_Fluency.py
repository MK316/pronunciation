import streamlit as st
import librosa
import numpy as np
import io
import plotly.graph_objects as go
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS

# --- 1. 페이지 설정 및 상태 초기화 ---
st.set_page_config(page_title="Step 1: Native Baseline Analyzer", layout="wide")

# 사이드바 리셋 버튼: 모든 데이터를 지우고 처음 상태로 돌아감
if st.sidebar.button("🔄 전체 초기화 (Reset All)"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

st.title("📊 Step 1: Native Baseline & Rhythm Analyzer")
st.markdown("원어민의 발화 리듬을 기준으로 여러분의 **연결성**과 **속도**를 정밀 분석합니다.")

target_sentence = "The quick brown fox jumps over the lazy dog."
st.info(f"**Read this:** {target_sentence}")

# --- [핵심] 원어민 기준점(Baseline) 생성 및 분석 함수 ---
@st.cache_resource # 앱 실행 시 한 번만 실행되도록 캐싱
def get_native_baseline(text):
    with st.spinner("원어민 기준점 생성 및 분석 중..."):
        # A. gTTS로 오디오 생성
        tts = gTTS(text=text, lang='en', tld='com') # 미국 영어 기준
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        # B. Librosa 분석을 위해 WAV로 변환
        audio_segment = AudioSegment.from_file(mp3_fp, format="mp3")
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        
        # C. 리듬 분석 (학생 분석 로직과 동일 적용)
        y, sr = librosa.load(wav_io, sr=None)
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        duration = librosa.get_duration(y=y_trimmed, sr=sr)
        
        rms = librosa.feature.rms(y=y_trimmed)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        frame_dur = duration / len(rms)
        threshold = -25 # 음절 간 끊김 타겟 까다로운 기준
        is_silent = rms_db < threshold
        
        silent_durations = []
        current_pause = 0
        for silent in is_silent:
            if silent: current_pause += frame_dur
            else:
                if current_pause > 0: silent_durations.append(current_pause)
                current_pause = 0
        
        short_gaps = [d for d in silent_durations if 0.05 <= d < 0.25]
        long_pauses = [d for d in silent_durations if d >= 0.25]
        
        word_count = len(text.split())
        baseline = {
            'y_trimmed': y_trimmed,
            'sr': sr,
            'duration': duration,
            'speech_rate': ((word_count * 1.3) / duration) * 60, # SPM 추정
            'staccato_count': len(short_gaps),
            'pause_count': len(long_pauses)
        }
        return baseline

# 앱 실행 시 원어민 베이스라인 자동 생성
if 'native_baseline' not in st.session_state:
    st.session_state.native_baseline = get_native_baseline(target_sentence)

# 사이드바에 원어민 발음 들어보기 추가
with st.sidebar:
    st.subheader("🎧 원어민 발음 듣기")
    # 캐싱된 오디오 데이터를 플레이어로 재생
    audio_segment_export = AudioSegment.from_mono_audiosegments(
        AudioSegment(
            st.session_state.native_baseline['y_trimmed'].tobytes(), 
            frame_rate=st.session_state.native_baseline['sr'],
            sample_width=st.session_state.native_baseline['y_trimmed'].dtype.itemsize, 
            channels=1
        )
    )
    play_io = io.BytesIO()
    audio_segment_export.export(play_io, format="wav")
    st.audio(play_io)

# --- 2. 학생 녹음 섹션 ---
col_rec, _ = st.columns([1, 5])
with col_rec:
    # key에 session_state를 연동하여 리셋 시 버튼도 초기화
    audio_data = mic_recorder(
        start_prompt="🔴 Start Recording",
        stop_prompt="⏹️ Stop & Analyze",
        key='rhythm_recorder_baseline_v1'
    )

# --- 3. 분석 및 대조 결과 출력 ---
if audio_data:
    try:
        with st.spinner("여러분의 발화 리듬 분석 중..."):
            audio_bytes = audio_data['bytes']
            
            # 유효성 검사 (너무 짧은 소음 방지)
            if len(audio_bytes) < 2000:
                st.warning("녹음 시간이 너무 짧습니다. 문장 전체를 다시 읽어주세요.")
                st.stop()

            # 오디오 로드 및 변환
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)

            y, sr = librosa.load(wav_io, sr=None)
            
            # Trimming (무음 제거)
            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
            total_duration = librosa.get_duration(y=y_trimmed, sr=sr)

            # --- 에너지 기반 리듬 분석 ---
            rms = librosa.feature.rms(y=y_trimmed)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            frame_dur = total_duration / len(rms)

            # 임계값 설정
            threshold = -25 
            is_silent = rms_db < threshold
            
            silent_durations = []
            current_pause = 0
            for silent in is_silent:
                if silent: current_pause += frame_dur
                else:
                    if current_pause > 0: silent_durations.append(current_pause)
                    current_pause = 0

            # 지표 분류: 학생 데이터
            student_short_gaps = [d for d in silent_durations if 0.05 <= d < 0.25]
            student_long_pauses = [d for d in silent_durations if d >= 0.25]

            student_staccato_count = len(student_short_gaps)
            student_pause_count = len(student_long_pauses)
            
            # 발화 속도 (SPM)
            word_count = len(target_sentence.split())
            student_speech_rate = ((word_count * 1.3) / total_duration) * 60

        # --- 4. [핵심] 원어민 대조 결과 대시보드 ---
        st.divider()
        st.subheader("🎯 원어민 대조 분석 결과")
        
        # 원어민 데이터 가져오기
        native = st.session_state.native_baseline
        
        m1, m2, m3 = st.columns(3)
        
        with m1:
            # 발화 속도 대조 (델타 값 표기)
            rate_delta = student_speech_rate - native['speech_rate']
            st.metric(
                label="발화 속도 (Native vs You)", 
                value=f"{student_speech_rate:.1f} SPM", 
                delta=f"{rate_delta:.1f} SPM (Target: {native['speech_rate']:.1f})",
                delta_color="normal" if rate_delta > -20 else "inverse" # 원어민보다 너무 느리면 빨간색
            )
            st.caption("Native Speaker: 130~150 SPM")
            
        with m2:
            # 음절 간 끊김 대조 (델타 값 표기 - 낮을수록 좋음)
            gap_delta = student_staccato_count - native['staccato_count']
            st.metric(
                label="음절 간 끊김 (Native vs You)", 
                value=f"{student_staccato_count} 회", 
                delta=f"{gap_delta} 회 (Target: {native['staccato_count']} 회)",
                delta_color="inverse" # 높을수록 나쁨 (빨간색)
            )
            st.caption(f"단어 사이 미세 단절 (0.05s~0.25s)")

        with m3:
            # 긴 멈춤 대조
            pause_delta = student_pause_count - native['pause_count']
            st.metric(
                label="긴 멈춤 (Native vs You)", 
                value=f"{student_pause_count} 회", 
                delta=f"{pause_delta} 회 (Target: {native['pause_count']} 회)",
                delta_color="inverse" # 높을수록 나쁨 (빨간색)
            )
            st.caption("0.25초 이상 멈춤")

        # 시각화: 대조 바 차트
        st.write("---")
        st.subheader("📊 리듬 지표 대조 그래프")
        categories = ['Speech Rate', 'Syllable Gaps', 'Long Pauses']
        
        fig = go.Figure(data=[
            go.Bar(name='Native Speaker', x=categories, y=[native['speech_rate'], native['staccato_count'], native['pause_count']], marker_color='#E0E0E0'),
            go.Bar(name='You', x=categories, y=[student_speech_rate, student_staccato_count, student_pause_count], marker_color='#2E7D32')
        ])
        # 차트 레이아웃 조정
        fig.update_layout(barmode='group', height=400)
        # 발화 속도는 숫자가 커야 좋고, Gaps/Pauses는 작아야 좋으므로 Y축을 이중으로 쓰거나 단위를 잘 맞추는게 좋지만, 여기선 직관적으로 group bar로 표시
        st.plotly_chart(fig, use_container_width=True)

        # --- 5. 분석 피드백 (수정) ---
        st.divider()
        st.subheader("💡 Expert Feedback")
        if student_staccato_count >= (word_count * 0.4):
            st.error("🚩 **음절마다 끊어 읽기가 많아 보입니다.**")
            st.write(f"원어민은 단어 사이를 미세하게 {native['staccato_count']}번만 끊었지만, 여러분은 {student_staccato_count}번 끊었습니다. 소리를 다음 단어까지 부드럽게 밀어내는 **연음(Linking)** 연습을 해보세요. 한 호흡에 문장 끝까지 간다는 느낌이 중요합니다.")
        elif student_pause_count > (native['pause_count'] + 1):
            st.warning("문장 중간에 호흡이 자주 끊깁니다. 의미 단위(Chunk)로 묶어서 읽어보세요.")
        elif student_speech_rate < (native['speech_rate'] - 30):
            st.info(f"원어민보다 발화 속도가 상당히 느립니다 ({native['speech_rate']:.1f} vs {student_speech_rate:.1f} SPM). 조금 더 자신감 있게 속도를 높여보세요.")
        else:
            st.success("단어 간 연결이 매끄럽고 리듬감이 원어민과 매우 흡사합니다! 훌륭합니다.")

        # 하단 리셋 버튼
        if st.button("🔄 다시 시도하기 (Reset Analysis)"):
            reset_all()

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다. 다시 녹음해 주세요. 오류 내용: {e}")

else:
    st.write("---")
    st.info("좌측 상단의 버튼을 눌러 녹음을 시작하세요. (문장 전체 발화 권장)")
