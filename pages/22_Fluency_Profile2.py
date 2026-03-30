import io
import numpy as np
import librosa
import streamlit as st
import plotly.graph_objects as go

from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS

st.set_page_config(page_title="Fluency Profile Analyzer", layout="wide")

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
# 2. Practice materials
# --------------------------------------------------
LEVELS = {
    "Level 1": {
        "label": "Easy",
        "target_sps": (2.8, 4.2),
        "sentences": [
            {"text": "I go to school every morning before class.", "syllables": 11},
            {"text": "She likes reading books at home after dinner.", "syllables": 10},
            {"text": "We study English together after lunch every day.", "syllables": 12},
        ],
    },
    "Level 2": {
        "label": "Intermediate",
        "target_sps": (3.1, 4.8),
        "sentences": [
            {"text": "My teacher gave us a short speaking task in class today.", "syllables": 13},
            {"text": "I usually practice English with my friends online at night.", "syllables": 14},
            {"text": "The class started late because of the heavy rain this morning.", "syllables": 14},
        ],
    },
    "Level 3": {
        "label": "Advanced",
        "target_sps": (3.4, 5.2),
        "sentences": [
            {"text": "Many students feel more confident after regular speaking practice each week.", "syllables": 18},
            {"text": "Learning a language takes time, patience, and steady effort over time.", "syllables": 18},
            {"text": "Clear pronunciation helps listeners understand the message more easily in class.", "syllables": 18},
        ],
    },
}


# --------------------------------------------------
# 3. Helpers
# --------------------------------------------------
def init_level_state(level_name):
    if level_name not in st.session_state.results:
        st.session_state.results[level_name] = {}


def reset_level_progress():
    level = st.session_state.selected_level
    st.session_state.results[level] = {}
    st.session_state.current_idx = 0
    st.session_state.widget_id += 1
    st.rerun()


def convert_to_wav(audio_bytes):
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io


def fluency_band(score):
    if score >= 85:
        return "Very smooth delivery"
    elif score >= 70:
        return "Fairly fluent with some hesitation"
    elif score >= 55:
        return "Developing fluency"
    else:
        return "Frequent interruption or unstable pace"


def score_speech_rate(speech_rate, low, high):
    target = (low + high) / 2
    dist = abs(speech_rate - target)
    score = max(0.0, 100.0 - dist * 42.0)

    # penalty if far outside recommended band
    if speech_rate < low - 0.5 or speech_rate > high + 0.5:
        score -= 10
    return max(0.0, min(100.0, score))


def score_articulation_rate(art_rate, low, high):
    # articulation rate is normally a bit faster than speech rate
    target = ((low + high) / 2) + 0.7
    dist = abs(art_rate - target)
    score = max(0.0, 100.0 - dist * 35.0)

    if art_rate < low or art_rate > high + 1.6:
        score -= 8
    return max(0.0, min(100.0, score))


def score_pause_ratio(pause_ratio):
    # stricter than before: lower is better
    if pause_ratio <= 0.08:
        return 100.0
    elif pause_ratio <= 0.14:
        return 85.0
    elif pause_ratio <= 0.20:
        return 70.0
    elif pause_ratio <= 0.26:
        return 55.0
    elif pause_ratio <= 0.33:
        return 35.0
    else:
        return 15.0


def score_mean_length_of_run(mlr_syllables):
    # stricter scaling
    if mlr_syllables >= 8.5:
        return 100.0
    elif mlr_syllables >= 6.5:
        return 82.0
    elif mlr_syllables >= 4.8:
        return 65.0
    elif mlr_syllables >= 3.2:
        return 45.0
    else:
        return 20.0


def interpret_measure(name, score):
    if score >= 85:
        level = "strong"
    elif score >= 70:
        level = "fair"
    elif score >= 55:
        level = "developing"
    else:
        level = "weak"

    explanations = {
        "Speech rate": {
            "strong": "Your overall pace is steady and appropriate.",
            "fair": "Your overall pace is acceptable, but not fully stable yet.",
            "developing": "Your overall pace needs more consistency.",
            "weak": "Your overall pace is quite unstable or far from the target range.",
        },
        "Articulation rate": {
            "strong": "When you are actually speaking, your speed is smooth and efficient.",
            "fair": "Your speaking speed is fairly good, but can be smoother.",
            "developing": "Your actual speaking speed is still somewhat uneven.",
            "weak": "Your speed during actual speech is quite unstable.",
        },
        "Pause ratio": {
            "strong": "Silence does not interrupt your delivery very much.",
            "fair": "There are some interruptions, but they do not dominate your speech.",
            "developing": "Pauses interrupt your delivery noticeably.",
            "weak": "Too much silence is breaking the flow of your message.",
        },
        "Mean length of run": {
            "strong": "You can continue speaking in fairly long chunks.",
            "fair": "You can produce some connected chunks, though not consistently.",
            "developing": "Your speaking chunks are still rather short.",
            "weak": "You stop too often to maintain a smooth run of speech.",
        },
    }
    return explanations[name][level]


def analyze_and_score(audio_bytes, syllable_count, target_sps_range, is_student=False):
    try:
        wav_io = convert_to_wav(audio_bytes)
        y, sr = librosa.load(wav_io, sr=16000)

        total_duration = librosa.get_duration(y=y, sr=sr)

        rms_mean = np.sqrt(np.mean(y**2))
        if is_student and rms_mean < 0.005:
            return None

        # Slightly less harsh speech detection
        intervals = librosa.effects.split(
            y,
            top_db=24,
            frame_length=2048,
            hop_length=256
        )

        # Keep shorter speech chunks than before
        min_seg_dur = 0.08
        filtered = []
        for start, end in intervals:
            seg_dur = (end - start) / sr
            if seg_dur >= min_seg_dur:
                filtered.append([start, end])

        if len(filtered) == 0:
            return None

        # Merge nearby chunks more generously
        merged = [filtered[0]]
        merge_gap = int(0.25 * sr)
        for start, end in filtered[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= merge_gap:
                merged[-1][1] = end
            else:
                merged.append([start, end])

        # If the last chunk is very near the previous one, attach it
        # This helps recover weak utterance endings
        if len(merged) >= 2:
            last_gap = merged[-1][0] - merged[-2][1]
            if last_gap <= int(0.35 * sr):
                merged[-2][1] = merged[-1][1]
                merged = merged[:-1]

        speech_start = merged[0][0]
        speech_end = merged[-1][1]

        y_utt = y[speech_start:speech_end]
        utterance_duration = librosa.get_duration(y=y_utt, sr=sr)

        if is_student and utterance_duration < 1.0:
            return None

        speech_only_duration = sum((end - start) / sr for start, end in merged)
        pause_time = max(0.0, utterance_duration - speech_only_duration)
        pause_ratio = pause_time / utterance_duration if utterance_duration > 0 else 0.0

        # Pause analysis inside utterance window
        rms = librosa.feature.rms(y=y_utt, frame_length=2048, hop_length=256)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        frame_dur = utterance_duration / len(rms) if len(rms) > 0 else 0
        is_silent = rms_db < -30

        pauses = []
        curr = 0.0
        for s in is_silent:
            if s:
                curr += frame_dur
            else:
                if curr > 0:
                    pauses.append(curr)
                curr = 0.0
        if curr > 0:
            pauses.append(curr)

        short_breaks = len([d for d in pauses if 0.05 <= d < 0.25])
        long_pauses = len([d for d in pauses if d >= 0.25])

        speech_rate = syllable_count / utterance_duration if utterance_duration > 0 else 0.0
        articulation_rate = syllable_count / speech_only_duration if speech_only_duration > 0 else 0.0

        weighted_breaks = short_breaks + (long_pauses * 1.5)
        n_runs = max(1.0, 1.0 + weighted_breaks)
        mean_length_of_run = syllable_count / n_runs

        low, high = target_sps_range

        profile_scores = {
            "Speech rate": score_speech_rate(speech_rate, low, high),
            "Articulation rate": score_articulation_rate(articulation_rate, low, high),
            "Pause ratio": score_pause_ratio(pause_ratio),
            "Mean length of run": score_mean_length_of_run(mean_length_of_run),
        }

        weighted_total = (
            profile_scores["Speech rate"] * 0.20 +
            profile_scores["Articulation rate"] * 0.20 +
            profile_scores["Pause ratio"] * 0.30 +
            profile_scores["Mean length of run"] * 0.30
        )

        hesitation_penalty = (short_breaks * 1.5) + (long_pauses * 5.0)

        rule_penalty = 0
        if pause_ratio >= 0.32:
            rule_penalty += 6
        elif pause_ratio >= 0.24:
            rule_penalty += 3

        if short_breaks >= 5:
            rule_penalty += 4
        elif short_breaks >= 3:
            rule_penalty += 2

        if long_pauses >= 3:
            rule_penalty += 8
        elif long_pauses >= 2:
            rule_penalty += 4

        total_score = max(0, round(weighted_total - hesitation_penalty - rule_penalty))

        leading_silence = speech_start / sr
        trailing_silence = total_duration - (speech_end / sr)

        return {
            "score": total_score,
            "wav": wav_io.getvalue(),
            "total_duration": total_duration,
            "utterance_duration": utterance_duration,
            "speech_only_duration": speech_only_duration,
            "leading_silence": leading_silence,
            "trailing_silence": trailing_silence,
            "syllable_count": syllable_count,
            "pause_list": pauses,
            "speech_intervals": [(s / sr, e / sr) for s, e in merged],
            "speech_start": speech_start / sr,
            "speech_end": speech_end / sr,
            "wave_y": y,
            "wave_sr": sr,
            "short_breaks": short_breaks,
            "long_pauses": long_pauses,
            "pause_ratio_raw": pause_ratio,
            "speech_rate": speech_rate,
            "articulation_rate": articulation_rate,
            "mean_length_of_run": mean_length_of_run,
            "profile_scores": profile_scores,
            "weighted_total_before_penalty": weighted_total,
            "hesitation_penalty": hesitation_penalty,
            "rule_penalty": rule_penalty,
        }

    except Exception:
        return None

@st.cache_resource
def get_native_audio_and_analysis(text, syllable_count, target_sps_range):
    tts = gTTS(text=text, lang="en", tld="com")
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_bytes = mp3_fp.getvalue()
    return analyze_and_score(
        mp3_bytes,
        syllable_count=syllable_count,
        target_sps_range=target_sps_range,
        is_student=False,
    )


def plot_waveform_with_marks(res):
    y = res["wave_y"]
    total_duration = res["total_duration"]
    speech_start = res["speech_start"]
    speech_end = res["speech_end"]
    speech_intervals = res["speech_intervals"]

    times = np.linspace(0, total_duration, num=len(y))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=y,
            mode="lines",
            name="Waveform",
            line=dict(width=1),
        )
    )

    fig.add_vrect(
        x0=speech_start,
        x1=speech_end,
        fillcolor="lightcoral",
        opacity=0.25,
        line_width=0,
        annotation_text="Analyzed speaking duration",
        annotation_position="top left",
    )

    for s, e in speech_intervals:
        fig.add_vrect(
            x0=s,
            x1=e,
            fillcolor="lightgreen",
            opacity=0.20,
            line_width=0,
        )

    fig.add_vline(x=speech_start, line_dash="dash")
    fig.add_vline(x=speech_end, line_dash="dash")

    fig.update_layout(
        title="Recorded waveform with analyzed speaking region",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=340,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def plot_radar_chart(profile_scores):
    labels = list(profile_scores.keys())
    values = list(profile_scores.values())

    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            name="Fluency profile",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title="Mini Fluency Profile",
        height=450,
        showlegend=False,
    )
    return fig


def describe_profile(res):
    lines = []
    for name in ["Speech rate", "Articulation rate", "Pause ratio", "Mean length of run"]:
        lines.append(f"**{name}**: {interpret_measure(name, res['profile_scores'][name])}")
    return "\n\n".join(lines)


def overall_feedback_from_profile(res):
    profile = res["profile_scores"]
    weakest = min(profile, key=profile.get)
    strongest = max(profile, key=profile.get)

    strengths = {
        "Speech rate": "Your overall pace is one of your stronger points.",
        "Articulation rate": "Your speed during actual speaking is one of your stronger points.",
        "Pause ratio": "You are controlling silence relatively well.",
        "Mean length of run": "You can maintain connected speech chunks relatively well.",
    }

    improvement = {
        "Speech rate": "Work on keeping a more stable overall pace from beginning to end.",
        "Articulation rate": "Try to keep the words moving more smoothly when you are actively speaking.",
        "Pause ratio": "Try to reduce unnecessary silence that interrupts the message.",
        "Mean length of run": "Try to continue speaking in longer chunks before stopping.",
    }

    return f"{strengths[strongest]} {improvement[weakest]}"


def overall_band_text(score):
    return fluency_band(score)


# --------------------------------------------------
# 4. UI
# --------------------------------------------------
st.title("📊 Fluency Profile Analyzer")
st.caption("Record 3 short sentences. The app creates a stricter mini fluency profile from four measures.")

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
    f"Recommended speech-rate range: {target_sps_range[0]} ~ {target_sps_range[1]} syllables per second"
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
        st.caption(
            f"Model speech rate: {native['speech_rate']:.2f} | "
            f"Model articulation rate: {native['articulation_rate']:.2f}"
        )

st.divider()

# --------------------------------------------------
# 5. Recorder
# --------------------------------------------------
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
        st.error("Voice was not detected clearly. Please record again.")

# --------------------------------------------------
# 6. Current trial result
# --------------------------------------------------
level_results = st.session_state.results[level_name]

if trial_num in level_results:
    res = level_results[trial_num]

    score_color = "green" if res["score"] >= 85 else "orange" if res["score"] >= 70 else "red"

    st.markdown(f"### Sentence {trial_num + 1} Fluency Score: :{score_color}[{res['score']} / 100]")
    st.progress(res["score"] / 100)
    st.caption(f"Interpretation: **{overall_band_text(res['score'])}**")

    st.audio(res["wav"], format="audio/wav")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Speech rate", f"{res['speech_rate']:.2f}")
    a2.metric("Articulation rate", f"{res['articulation_rate']:.2f}")
    a3.metric("Pause ratio", f"{res['pause_ratio_raw']:.2f}")
    a4.metric("Mean length of run", f"{res['mean_length_of_run']:.2f}")

    b1, b2, b3 = st.columns(3)
    b1.metric("Recorded duration", f"{res['total_duration']:.2f} sec")
    b2.metric("Utterance window", f"{res['utterance_duration']:.2f} sec")
    b3.metric("Speech-only duration", f"{res['speech_only_duration']:.2f} sec")

    st.plotly_chart(plot_waveform_with_marks(res), use_container_width=True)
    st.plotly_chart(plot_radar_chart(res["profile_scores"]), use_container_width=True)

    st.subheader("What this mini profile means")
    st.markdown(
        """
- **Speech rate** = your overall pace across the sentence  
- **Articulation rate** = your speed during actual speaking  
- **Pause ratio** = how much silence interrupts your delivery  
- **Mean length of run** = how long you continue smoothly before a major break  
"""
    )
    st.markdown(describe_profile(res))
    st.info(overall_feedback_from_profile(res))

    with st.expander("See detailed values"):
        st.write(f"Preset syllable count: {res['syllable_count']}")
        st.write(f"Short breaks: {res['short_breaks']}")
        st.write(f"Long pauses: {res['long_pauses']}")
        st.write(f"Leading silence: {res['leading_silence']:.2f} sec")
        st.write(f"Trailing silence: {res['trailing_silence']:.2f} sec")
        st.write(f"Weighted score before penalties: {res['weighted_total_before_penalty']:.1f}")
        st.write(f"Hesitation penalty: {res['hesitation_penalty']:.1f}")
        st.write(f"Rule penalty: {res['rule_penalty']}")
        st.write("Profile scores:")
        st.json({k: round(v, 1) for k, v in res["profile_scores"].items()})

st.divider()

# --------------------------------------------------
# 7. Next sentence
# --------------------------------------------------
if trial_num < 2:
    if trial_num in level_results:
        if st.button("➡️ Go to next sentence"):
            st.session_state.current_idx += 1
            st.session_state.widget_id += 1
            st.rerun()
    else:
        st.caption("Record this sentence first to move to the next one.")

# --------------------------------------------------
# 8. Final average result
# --------------------------------------------------
if len(level_results) == 3:
    st.divider()
    st.subheader("🏆 Final Average Result")

    ordered_results = [level_results[i] for i in range(3)]

    avg_score = float(np.mean([r["score"] for r in ordered_results]))
    avg_speech_rate = float(np.mean([r["speech_rate"] for r in ordered_results]))
    avg_articulation_rate = float(np.mean([r["articulation_rate"] for r in ordered_results]))
    avg_pause_ratio = float(np.mean([r["pause_ratio_raw"] for r in ordered_results]))
    avg_mlr = float(np.mean([r["mean_length_of_run"] for r in ordered_results]))

    avg_profile_scores = {
        "Speech rate": float(np.mean([r["profile_scores"]["Speech rate"] for r in ordered_results])),
        "Articulation rate": float(np.mean([r["profile_scores"]["Articulation rate"] for r in ordered_results])),
        "Pause ratio": float(np.mean([r["profile_scores"]["Pause ratio"] for r in ordered_results])),
        "Mean length of run": float(np.mean([r["profile_scores"]["Mean length of run"] for r in ordered_results])),
    }

    final_color = "green" if avg_score >= 85 else "orange" if avg_score >= 70 else "red"

    st.markdown(f"### Average Fluency Score: :{final_color}[{avg_score:.1f} / 100]")
    st.progress(avg_score / 100)
    st.caption(f"Interpretation: **{overall_band_text(avg_score)}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg speech rate", f"{avg_speech_rate:.2f}")
    c2.metric("Avg articulation rate", f"{avg_articulation_rate:.2f}")
    c3.metric("Avg pause ratio", f"{avg_pause_ratio:.2f}")
    c4.metric("Avg mean length of run", f"{avg_mlr:.2f}")

    st.plotly_chart(plot_radar_chart(avg_profile_scores), use_container_width=True)

    weakest = min(avg_profile_scores, key=avg_profile_scores.get)
    strongest = max(avg_profile_scores, key=avg_profile_scores.get)

    st.subheader("How to read your final profile")
    st.markdown(
        f"""
- **Speech rate** = your overall pace  
- **Articulation rate** = your speed during actual speaking  
- **Pause ratio** = how much silence interrupts your delivery  
- **Mean length of run** = how long you continue smoothly before a major break  

Your strongest area is **{strongest}**.  
The area that needs the most attention now is **{weakest}**.
"""
    )

    if st.button("🔁 Start this level again"):
        reset_level_progress()
