import io
import numpy as np
import librosa
import streamlit as st
import plotly.graph_objects as go

from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS

st.set_page_config(page_title="Fluency Score Analyzer", layout="wide")

# --------------------------------------------------
# 1. Session state
# --------------------------------------------------
if "widget_id" not in st.session_state:
    st.session_state.widget_id = 0

if "selected_level" not in st.session_state:
    st.session_state.selected_level = "Level 1"

if "results" not in st.session_state:
    st.session_state.results = {}

if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0


# --------------------------------------------------
# 2. Practice materials by level
# --------------------------------------------------
LEVELS = {
    "Level 1": {
        "label": "Easy",
        "target_sps": (2.8, 4.5),
        "sentences": [
            {"text": "I go to school every morning.", "syllables": 9},
            {"text": "She likes reading books at home.", "syllables": 8},
            {"text": "We study English after lunch.", "syllables": 8},
        ],
    },
    "Level 2": {
        "label": "Intermediate",
        "target_sps": (3.2, 5.0),
        "sentences": [
            {"text": "My teacher gave us a short speaking task today.", "syllables": 11},
            {"text": "I usually practice English with my friends online.", "syllables": 12},
            {"text": "The class started late because of the heavy rain.", "syllables": 11},
        ],
    },
    "Level 3": {
        "label": "Advanced",
        "target_sps": (3.5, 5.5),
        "sentences": [
            {"text": "Many students feel more confident after regular speaking practice.", "syllables": 16},
            {"text": "Learning a language takes time, patience, and daily effort.", "syllables": 16},
            {"text": "Clear pronunciation helps listeners understand the message better.", "syllables": 16},
        ],
    },
}


# --------------------------------------------------
# 3. Helpers
# --------------------------------------------------
def reset_level_progress():
    level = st.session_state.selected_level
    st.session_state.results[level] = {}
    st.session_state.current_idx = 0
    st.session_state.widget_id += 1
    st.rerun()


def init_level_state(level_name):
    if level_name not in st.session_state.results:
        st.session_state.results[level_name] = {}


def convert_to_wav(audio_bytes):
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io


def analyze_and_score(audio_bytes, syllable_count, target_sps_range, is_student=False):
    try:
        wav_io = convert_to_wav(audio_bytes)
        y, sr = librosa.load(wav_io, sr=16000)

        # Basic silence / noise guard
        rms_mean = np.sqrt(np.mean(y**2))
        if is_student and rms_mean < 0.01:
            return None

        y = librosa.util.normalize(y)
        y_trim, _ = librosa.effects.trim(y, top_db=30)
        duration = librosa.get_duration(y=y_trim, sr=sr)

        if is_student and duration < 1.2:
            return None

        # RMS-based pause estimate
        rms = librosa.feature.rms(y=y_trim)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        frame_dur = duration / len(rms) if len(rms) > 0 else 0
        is_silent = rms_db < -30

        pauses = []
        curr = 0
        for s in is_silent:
            if s:
                curr += frame_dur
            else:
                if curr > 0:
                    pauses.append(curr)
                curr = 0
        if curr > 0:
            pauses.append(curr)

        staccato = len([d for d in pauses if 0.05 <= d < 0.25])
        long_pause = len([d for d in pauses if d >= 0.25])

        # SPS / SPM
        sps = syllable_count / duration if duration > 0 else 0
        rate_spm = sps * 60

        # Scoring
        low, high = target_sps_range

        # Speed score (60)
        if low <= sps <= high:
            speed_score = 60
        else:
            dist = min(abs(sps - low), abs(sps - high))
            speed_score = max(20, 60 - (dist * 20))

        # Connectivity score (40)
        conn_score = max(0, 40 - (staccato * 5) - (long_pause * 10))

        total_score = int(round(speed_score + conn_score))

        return {
            "rate": rate_spm,
            "sps": sps,
            "staccato": staccato,
            "pause": long_pause,
            "score": total_score,
            "wav": wav_io.getvalue(),
            "dur": duration,
        }

    except Exception:
        return None


@st.cache_resource
def get_native_audio_and_analysis(text, syllable_count, target_sps_range):
    tts = gTTS(text=text, lang="en", tld="com")
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_bytes = mp3_fp.getvalue()
    native_res = analyze_and_score(
        mp3_bytes,
        syllable_count=syllable_count,
        target_sps_range=target_sps_range,
        is_student=False,
    )
    return native_res


def trial_feedback(res, target_sps_range):
    low, high = target_sps_range

    if res["score"] >= 90:
        return "Excellent fluency. Your speed and connected speech are both very strong."
    elif low <= res["sps"] <= high:
        if res["pause"] > 1:
            return "Your speaking rate is appropriate. Try to reduce long pauses for smoother delivery."
        elif res["staccato"] > 2:
            return "Your speed is appropriate, but the rhythm sounds slightly broken. Focus on smoother chunking."
        else:
            return "Good job. Your speed is stable and your speech is fairly connected."
    else:
        if res["sps"] < low:
            return "You are speaking a bit slowly. Try to keep the sentence flowing without stopping too often."
        else:
            return "You are speaking a bit too fast. Try to keep a steady rhythm and clearer phrasing."


def overall_feedback(avg_score, avg_sps, avg_staccato, avg_pause, target_sps_range):
    low, high = target_sps_range

    if avg_score >= 90:
        return "Outstanding overall fluency across all three sentences."
    elif avg_score >= 80:
        return "Very good overall fluency. Your rhythm is mostly stable across the trials."
    elif avg_score >= 70:
        if avg_sps < low:
            return "Your performance is fairly consistent, but your overall speed is slightly slow. Practice speaking in larger chunks."
        elif avg_sps > high:
            return "Your performance is fairly consistent, but your overall speed is slightly fast. Slow down a little for clearer phrasing."
        elif avg_pause >= 2:
            return "Your speed is acceptable overall, but reducing long pauses will improve your fluency."
        else:
            return "Good progress. More repetition will help make your speech smoother and more natural."
    else:
        return "You need more practice to build stable fluency. Focus on a steady pace and fewer pauses."


# --------------------------------------------------
# 4. UI header
# --------------------------------------------------
st.title("📊 Speaking Rate Analyzer")
st.caption("Record 3 short sentences. Your final result is based on the average of all 3 trials.")

level_name = st.selectbox(
    "Choose your level",
    list(LEVELS.keys()),
    index=list(LEVELS.keys()).index(st.session_state.selected_level),
)

if level_name != st.session_state.selected_level:
    st.session_state.selected_level = level_name
    st.session_state.current_idx = 0
    st.session_state.widget_id += 1

init_level_state(level_name)
level_data = LEVELS[level_name]
sentences = level_data["sentences"]
target_sps_range = level_data["target_sps"]

st.info(
    f"**Selected level:** {level_name} ({level_data['label']})  \n"
    f"**Recommended speed:** {target_sps_range[0]} ~ {target_sps_range[1]} SPS"
)

trial_num = st.session_state.current_idx
current_item = sentences[trial_num]
current_text = current_item["text"]
current_syllables = current_item["syllables"]

st.subheader(f"Sentence {trial_num + 1} of 3")
st.markdown(f"**Target sentence:** {current_text}")

native = get_native_audio_and_analysis(
    current_text,
    syllable_count=current_syllables,
    target_sps_range=target_sps_range,
)

with st.expander("🎧 Listen to the model sentence", expanded=True):
    if native:
        st.audio(native["wav"], format="audio/wav")

# --------------------------------------------------
# 5. Recorder
# --------------------------------------------------
st.divider()

col1, col2, col3 = st.columns([1.5, 1, 1])
with col1:
    audio_data = mic_recorder(
        start_prompt="🔴 Start recording",
        stop_prompt="⏹️ Stop and analyze",
        key=f"rec_{level_name}_{trial_num}_{st.session_state.widget_id}",
    )
with col2:
    if st.button("⬅️ Previous", disabled=(trial_num == 0)):
        st.session_state.current_idx -= 1
        st.session_state.widget_id += 1
        st.rerun()
with col3:
    if st.button("🔄 Reset this level"):
        reset_level_progress()

if audio_data:
    result = analyze_and_score(
        audio_data["bytes"],
        syllable_count=current_syllables,
        target_sps_range=target_sps_range,
        is_student=True,
    )
    if result:
        st.session_state.results[level_name][trial_num] = result
        st.success(f"Sentence {trial_num + 1} has been analyzed.")
    else:
        st.error("⚠️ Voice was not detected clearly. Please record again.")

# --------------------------------------------------
# 6. Current trial result
# --------------------------------------------------
level_results = st.session_state.results[level_name]

if trial_num in level_results:
    res = level_results[trial_num]

    score_color = "green" if res["score"] >= 80 else "orange" if res["score"] >= 60 else "red"

    st.markdown(f"### Sentence {trial_num + 1} Score: :{score_color}[{res['score']} / 100]")
    st.progress(res["score"] / 100)

    st.audio(res["wav"], format="audio/wav")

    m1, m2, m3 = st.columns(3)
    m1.metric("SPS", f"{res['sps']:.2f}", delta=f"Target: {target_sps_range[0]}~{target_sps_range[1]}")
    m2.metric("Short breaks", f"{res['staccato']}", delta_color="inverse")
    m3.metric("Long pauses", f"{res['pause']}", delta_color="inverse")

    st.write("**Feedback**")
    st.info(trial_feedback(res, target_sps_range))

# --------------------------------------------------
# 7. Navigation to next sentence
# --------------------------------------------------
st.divider()

if trial_num < 2:
    if trial_num in level_results:
        if st.button("➡️ Go to next sentence"):
            st.session_state.current_idx += 1
            st.session_state.widget_id += 1
            st.rerun()
    else:
        st.caption("Record this sentence first to move to the next one.")

# --------------------------------------------------
# 8. Final summary after 3 trials
# --------------------------------------------------
if len(level_results) == 3:
    st.divider()
    st.subheader("🏆 Final Average Result")

    ordered_results = [level_results[i] for i in range(3)]

    avg_score = float(np.mean([r["score"] for r in ordered_results]))
    avg_sps = float(np.mean([r["sps"] for r in ordered_results]))
    avg_staccato = float(np.mean([r["staccato"] for r in ordered_results]))
    avg_pause = float(np.mean([r["pause"] for r in ordered_results]))

    final_color = "green" if avg_score >= 80 else "orange" if avg_score >= 60 else "red"

    st.markdown(f"### Average Fluency Score: :{final_color}[{avg_score:.1f} / 100]")
    st.progress(avg_score / 100)

    c1, c2, c3 = st.columns(3)
    c1.metric("Average SPS", f"{avg_sps:.2f}")
    c2.metric("Average short breaks", f"{avg_staccato:.2f}")
    c3.metric("Average long pauses", f"{avg_pause:.2f}")

    # Trial score chart
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Sentence 1", "Sentence 2", "Sentence 3"],
            y=[ordered_results[0]["score"], ordered_results[1]["score"], ordered_results[2]["score"]],
            name="Trial score",
        )
    )
    fig.add_hline(y=avg_score, line_dash="dash", annotation_text=f"Average: {avg_score:.1f}")
    fig.update_layout(
        title="Scores across the 3 sentences",
        yaxis_title="Score",
        xaxis_title="Sentence",
        yaxis_range=[0, 100],
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("**Overall feedback**")
    st.success(overall_feedback(avg_score, avg_sps, avg_staccato, avg_pause, target_sps_range))

    with st.expander("See sentence-by-sentence summary"):
        for i, r in enumerate(ordered_results, start=1):
            st.markdown(
                f"""
**Sentence {i}**
- Score: {r['score']}
- SPS: {r['sps']:.2f}
- Short breaks: {r['staccato']}
- Long pauses: {r['pause']}
"""
            )

    if st.button("🔁 Start this level again"):
        reset_level_progress()
