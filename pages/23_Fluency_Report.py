import io
import math
import tempfile
from datetime import datetime
import textwrap
import numpy as np
import librosa
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from pydub import AudioSegment
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

st.set_page_config(page_title="Fluency Profile Analyzer", layout="wide")

# --------------------------------------------------
# 1. Session state
# --------------------------------------------------
DEFAULTS = {
    "widget_id": 0,
    "selected_level": "Level 1",
    "results": {},
    "current_idx": 0,
    "session_started": False,
    "user_name": "",
    "session_start_time": None,
    "raw_recordings": {},
    "manual_cuts": {},
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


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
    if level_name not in st.session_state.raw_recordings:
        st.session_state.raw_recordings[level_name] = {}
    if level_name not in st.session_state.manual_cuts:
        st.session_state.manual_cuts[level_name] = {}


def reset_level_progress():
    level = st.session_state.selected_level
    st.session_state.results[level] = {}
    st.session_state.raw_recordings[level] = {}
    st.session_state.manual_cuts[level] = {}
    st.session_state.current_idx = 0
    st.session_state.widget_id += 1
    st.rerun()


def start_new_session(user_name):
    st.session_state.session_started = True
    st.session_state.user_name = user_name.strip()
    st.session_state.session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.results = {}
    st.session_state.raw_recordings = {}
    st.session_state.manual_cuts = {}
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


def load_audio_array(audio_bytes):
    wav_io = convert_to_wav(audio_bytes)
    y, sr = librosa.load(wav_io, sr=16000)
    wav_bytes = wav_io.getvalue()
    return y, sr, wav_bytes


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
    if speech_rate < low - 0.5 or speech_rate > high + 0.5:
        score -= 10
    return max(0.0, min(100.0, score))


def score_articulation_rate(art_rate, low, high):
    target = ((low + high) / 2) + 0.7
    dist = abs(art_rate - target)
    score = max(0.0, 100.0 - dist * 35.0)
    if art_rate < low or art_rate > high + 1.6:
        score -= 8
    return max(0.0, min(100.0, score))


def score_pause_ratio(pause_ratio):
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


def analyze_selected_segment(y_full, sr, start_sec, end_sec, syllable_count, target_sps_range, wav_bytes):
    start_sample = max(0, int(start_sec * sr))
    end_sample = min(len(y_full), int(end_sec * sr))

    if end_sample <= start_sample:
        return None

    y_utt = y_full[start_sample:end_sample]
    utterance_duration = librosa.get_duration(y=y_utt, sr=sr)

    if utterance_duration < 0.8:
        return None

    intervals = librosa.effects.split(
        y_utt,
        top_db=24,
        frame_length=2048,
        hop_length=256
    )

    min_seg_dur = 0.08
    filtered = []
    for s, e in intervals:
        seg_dur = (e - s) / sr
        if seg_dur >= min_seg_dur:
            filtered.append([s, e])

    if len(filtered) == 0:
        return None

    merged = [filtered[0]]
    merge_gap = int(0.25 * sr)
    for s, e in filtered[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    if len(merged) >= 2:
        last_gap = merged[-1][0] - merged[-2][1]
        if last_gap <= int(0.35 * sr):
            merged[-2][1] = merged[-1][1]
            merged = merged[:-1]

    speech_only_duration = sum((e - s) / sr for s, e in merged)
    pause_time = max(0.0, utterance_duration - speech_only_duration)
    pause_ratio = pause_time / utterance_duration if utterance_duration > 0 else 0.0

    rms = librosa.feature.rms(y=y_utt, frame_length=2048, hop_length=256)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    frame_dur = utterance_duration / len(rms) if len(rms) > 0 else 0.0
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

    speech_intervals_global = [((s + start_sample) / sr, (e + start_sample) / sr) for s, e in merged]

    return {
        "score": total_score,
        "wav": wav_bytes,
        "wave_y": y_full,
        "wave_sr": sr,
        "total_duration": librosa.get_duration(y=y_full, sr=sr),
        "utterance_duration": utterance_duration,
        "speech_only_duration": speech_only_duration,
        "selected_start": start_sec,
        "selected_end": end_sec,
        "leading_silence": start_sec,
        "trailing_silence": max(0.0, librosa.get_duration(y=y_full, sr=sr) - end_sec),
        "syllable_count": syllable_count,
        "pause_list": pauses,
        "speech_intervals": speech_intervals_global,
        "speech_start": start_sec,
        "speech_end": end_sec,
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


@st.cache_resource
def get_native_audio_and_analysis(text, syllable_count, target_sps_range):
    tts = gTTS(text=text, lang="en", tld="com")
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_bytes = mp3_fp.getvalue()
    y, sr, wav_bytes = load_audio_array(mp3_bytes)
    total_dur = librosa.get_duration(y=y, sr=sr)
    return analyze_selected_segment(
        y, sr, 0.0, total_dur, syllable_count, target_sps_range, wav_bytes
    )


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


def plot_waveform_with_selection(y, sr, selected_start=None, selected_end=None, speech_intervals=None, title="Recorded waveform"):
    total_duration = librosa.get_duration(y=y, sr=sr)
    times = np.linspace(0, total_duration, num=len(y))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=y,
            mode="lines",
            line=dict(width=1),
            name="Waveform",
        )
    )

    if selected_start is not None and selected_end is not None:
        fig.add_vrect(
            x0=selected_start,
            x1=selected_end,
            fillcolor="lightcoral",
            opacity=0.25,
            line_width=0,
            annotation_text="Selected utterance",
            annotation_position="top left",
        )
        fig.add_vline(x=selected_start, line_dash="dash")
        fig.add_vline(x=selected_end, line_dash="dash")

    if speech_intervals:
        for s, e in speech_intervals:
            fig.add_vrect(
                x0=s,
                x1=e,
                fillcolor="lightgreen",
                opacity=0.18,
                line_width=0,
            )

    fig.update_layout(
        title=title,
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
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Mini Fluency Profile",
        height=430,
        showlegend=False,
    )
    return fig


def make_radar_png(profile_scores):
    labels = list(profile_scores.keys())
    values = list(profile_scores.values())
    N = len(labels)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles_closed, values_closed, linewidth=2)
    ax.fill(angles_closed, values_closed, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_title("Final Fluency Profile", pad=20)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def build_pdf_report(user_name, start_time, end_time, level_name, avg_score, avg_profile_scores, avg_metrics, band_text):
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4

    margin_x = 18 * mm
    margin_y = 18 * mm
    y = height - 20 * mm

    c.setTitle("Fluency Feedback Report")

    # ---------------------------
    # Header
    # ---------------------------
    c.setFont("Helvetica-Bold", 17)
    c.drawString(margin_x, y, "Fluency Feedback Report")
    y -= 10 * mm

    c.setStrokeColor(colors.grey)
    c.line(margin_x, y, width - margin_x, y)
    y -= 8 * mm

    c.setFont("Helvetica", 11)
    c.drawString(margin_x, y, f"Name: {user_name}")
    y -= 6 * mm
    c.drawString(margin_x, y, f"Level: {level_name}")
    y -= 6 * mm
    c.drawString(margin_x, y, f"Session start time: {start_time}")
    y -= 6 * mm
    c.drawString(margin_x, y, f"Report generated time: {end_time}")
    y -= 10 * mm

    # ---------------------------
    # Average result
    # ---------------------------
    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin_x, y, "Average result")
    y -= 8 * mm

    c.setFont("Helvetica", 11)
    c.drawString(margin_x, y, f"Average fluency score: {avg_score:.1f} / 100")
    y -= 6 * mm
    c.drawString(margin_x, y, f"Interpretation: {band_text}")
    y -= 10 * mm

    # ---------------------------
    # Mini profile + radar chart side by side
    # ---------------------------
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_x, y, "Mini profile")
    y -= 8 * mm

    content_top = y

    # Left area: 1 x 4 vertical table
    left_x = margin_x
    left_w = 78 * mm
    row_h = 16 * mm
    table_top = content_top

    table_rows = [
        ("Speech rate", "Expected: 2.8-5.2", f"{avg_metrics['speech_rate']:.2f}"),
        ("Articulation rate", "Expected: 3.5-6.0", f"{avg_metrics['articulation_rate']:.2f}"),
        ("Pause ratio", "Expected: lower is better", f"{avg_metrics['pause_ratio']:.2f}"),
        ("Mean length of run", "Expected: 6.0+ desirable", f"{avg_metrics['mlr']:.2f}"),
    ]

    # Table border
    c.setStrokeColor(colors.black)
    c.rect(left_x, table_top - 4 * row_h, left_w, 4 * row_h, stroke=1, fill=0)
    for i in range(1, 4):
        y_line = table_top - i * row_h
        c.line(left_x, y_line, left_x + left_w, y_line)

    # Fill table
    pad_x = 4 * mm
    for i, (title, expected, value) in enumerate(table_rows):
        cell_top = table_top - i * row_h - 4 * mm
        c.setFont("Helvetica-Bold", 10)
        c.drawString(left_x + pad_x, cell_top, title)

        c.setFont("Helvetica", 8.8)
        c.drawString(left_x + pad_x, cell_top - 4.5 * mm, expected)

        c.setFont("Helvetica-Bold", 11.5)
        c.drawString(left_x + pad_x, cell_top - 10 * mm, value)

    # Right area: radar chart
    radar_img = make_radar_png(avg_profile_scores)
    img_reader = ImageReader(radar_img)

    chart_w = 82 * mm
    chart_h = 82 * mm
    chart_x = width - margin_x - chart_w
    chart_y = content_top - chart_h + 2 * mm

    c.drawImage(
        img_reader,
        chart_x,
        chart_y,
        width=chart_w,
        height=chart_h,
        preserveAspectRatio=True,
        mask="auto"
    )

    # Move y below both blocks
    block_bottom = min(table_top - 4 * row_h, chart_y)
    y = block_bottom - 10 * mm

    # ---------------------------
    # How to read this profile
    # ---------------------------
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_x, y, "How to read this profile")
    y -= 8 * mm

    explain_lines = [
        "Speech rate shows the overall pace across the full selected utterance.",
        "Articulation rate shows how fast the speaker moves during actual speech, excluding silent intervals.",
        "Pause ratio shows how much silence interrupts delivery. Lower values are better.",
        "Mean length of run shows how long the speaker can continue smoothly before a noticeable break.",
    ]

    c.setFont("Helvetica", 10.5)
    for paragraph in explain_lines:
        wrapped = textwrap.wrap(paragraph, width=92)
        for line in wrapped:
            c.drawString(margin_x, y, line)
            y -= 5.2 * mm
        y -= 1 * mm

    y -= 2 * mm

    # ---------------------------
    # Profile scores
    # ---------------------------
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_x, y, "Profile scores")
    y -= 8 * mm

    c.setFont("Helvetica", 10.5)
    for k in ["Speech rate", "Articulation rate", "Pause ratio", "Mean length of run"]:
        c.drawString(margin_x, y, f"{k}: {avg_profile_scores[k]:.1f}")
        y -= 5.2 * mm

    c.showPage()
    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
# --------------------------------------------------
# 4. App header and start gate
# --------------------------------------------------
st.title("📊 Fluency Profile Analyzer")
st.caption("Record 3 sentences, trim the utterance manually, analyze the selected part, and generate a PDF report at the end.")

if not st.session_state.session_started:
    st.subheader("Start session")
    name_input = st.text_input("Enter user name")
    if st.button("Start Session"):
        if not name_input.strip():
            st.error("Please enter the user name first.")
        else:
            start_new_session(name_input)
    st.stop()

st.success(f"Session started for: {st.session_state.user_name}")
st.caption(f"Session start time: {st.session_state.session_start_time}")

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
    f"Selected level: {level_name} ({level_data['label']})\n\n"
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

with st.expander("Listen to the model sentence", expanded=True):
    if native:
        st.audio(native["wav"], format="audio/wav")
        st.caption(
            f"Model speech rate: {native['speech_rate']:.2f} | "
            f"Model articulation rate: {native['articulation_rate']:.2f}"
        )

st.divider()

# --------------------------------------------------
# 5. Recording
# --------------------------------------------------
col1, col2, col3 = st.columns([1.5, 1, 1])

with col1:
    audio_data = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        key=f"rec_{level_name}_{trial_num}_{st.session_state.widget_id}",
    )

with col2:
    if st.button("Previous", disabled=(trial_num == 0)):
        st.session_state.current_idx -= 1
        st.session_state.widget_id += 1
        st.rerun()

with col3:
    if st.button("Reset this level"):
        reset_level_progress()

if audio_data:
    y, sr, wav_bytes = load_audio_array(audio_data["bytes"])
    st.session_state.raw_recordings[level_name][trial_num] = {
        "y": y,
        "sr": sr,
        "wav": wav_bytes,
        "total_duration": librosa.get_duration(y=y, sr=sr),
    }
    st.success("Recording saved. Now select the beginning and ending of the utterance.")

# --------------------------------------------------
# 6. Manual selection and analysis
# --------------------------------------------------
raws = st.session_state.raw_recordings[level_name]
level_results = st.session_state.results[level_name]

if trial_num in raws:
    raw = raws[trial_num]
    total_duration = raw["total_duration"]

    default_start = 0.0
    default_end = total_duration

    if trial_num in st.session_state.manual_cuts[level_name]:
        default_start = st.session_state.manual_cuts[level_name][trial_num]["start"]
        default_end = st.session_state.manual_cuts[level_name][trial_num]["end"]

    st.subheader("Select the utterance range")
    st.plotly_chart(
        plot_waveform_with_selection(
            raw["y"], raw["sr"], default_start, default_end, title="Recorded waveform"
        ),
        use_container_width=True,
    )

    s1, s2 = st.columns(2)
    with s1:
        selected_start = st.slider(
            "Beginning of utterance (seconds)",
            min_value=0.0,
            max_value=float(round(total_duration, 2)),
            value=float(round(default_start, 2)),
            step=0.01,
            key=f"start_slider_{level_name}_{trial_num}",
        )
    with s2:
        selected_end = st.slider(
            "Ending of utterance (seconds)",
            min_value=0.0,
            max_value=float(round(total_duration, 2)),
            value=float(round(default_end, 2)),
            step=0.01,
            key=f"end_slider_{level_name}_{trial_num}",
        )

    st.session_state.manual_cuts[level_name][trial_num] = {
        "start": selected_start,
        "end": selected_end,
    }

    st.plotly_chart(
        plot_waveform_with_selection(
            raw["y"],
            raw["sr"],
            selected_start,
            selected_end,
            title="Waveform with selected utterance range",
        ),
        use_container_width=True,
    )

    if st.button("Analyze and show feedback", key=f"analyze_btn_{level_name}_{trial_num}"):
        if selected_end <= selected_start:
            st.error("The ending point must be later than the beginning point.")
        else:
            result = analyze_selected_segment(
                raw["y"],
                raw["sr"],
                selected_start,
                selected_end,
                current_syllables,
                target_sps_range,
                raw["wav"],
            )
            if result:
                st.session_state.results[level_name][trial_num] = result
                st.success("Analysis completed.")
            else:
                st.error("The selected part is too short or speech was not detected clearly.")

# --------------------------------------------------
# 7. Current result display
# --------------------------------------------------
if trial_num in level_results:
    res = level_results[trial_num]

    score_color = "green" if res["score"] >= 85 else "orange" if res["score"] >= 70 else "red"

    st.divider()
    st.markdown(f"### Sentence {trial_num + 1} Fluency Score: :{score_color}[{res['score']} / 100]")
    st.progress(res["score"] / 100)
    st.caption(f"Interpretation: **{fluency_band(res['score'])}**")

    st.audio(res["wav"], format="audio/wav")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Speech rate", f"{res['speech_rate']:.2f}")
    a2.metric("Articulation rate", f"{res['articulation_rate']:.2f}")
    a3.metric("Pause ratio", f"{res['pause_ratio_raw']:.2f}")
    a4.metric("Mean length of run", f"{res['mean_length_of_run']:.2f}")

    b1, b2, b3 = st.columns(3)
    b1.metric("Recorded duration", f"{res['total_duration']:.2f} sec")
    b2.metric("Selected utterance", f"{res['utterance_duration']:.2f} sec")
    b3.metric("Speech-only duration", f"{res['speech_only_duration']:.2f} sec")

    st.plotly_chart(
        plot_waveform_with_selection(
            res["wave_y"],
            res["wave_sr"],
            res["selected_start"],
            res["selected_end"],
            speech_intervals=res["speech_intervals"],
            title="Recorded waveform with selected utterance and detected speech",
        ),
        use_container_width=True,
    )

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
        st.write(f"Selected start: {res['selected_start']:.2f} sec")
        st.write(f"Selected end: {res['selected_end']:.2f} sec")
        st.write(f"Short breaks: {res['short_breaks']}")
        st.write(f"Long pauses: {res['long_pauses']}")
        st.write(f"Weighted score before penalties: {res['weighted_total_before_penalty']:.1f}")
        st.write(f"Hesitation penalty: {res['hesitation_penalty']:.1f}")
        st.write(f"Rule penalty: {res['rule_penalty']}")

st.divider()

# --------------------------------------------------
# 8. Next sentence
# --------------------------------------------------
if trial_num < 2:
    if trial_num in level_results:
        if st.button("Go to next sentence"):
            st.session_state.current_idx += 1
            st.session_state.widget_id += 1
            st.rerun()
    else:
        st.caption("Record the sentence, choose the utterance range, and analyze it first.")

# --------------------------------------------------
# 9. Final result and PDF
# --------------------------------------------------
if len(level_results) == 3:
    st.divider()
    st.subheader("Final Average Result")

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
    st.caption(f"Interpretation: **{fluency_band(avg_score)}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg speech rate", f"{avg_speech_rate:.2f}")
    c2.metric("Avg articulation rate", f"{avg_articulation_rate:.2f}")
    c3.metric("Avg pause ratio", f"{avg_pause_ratio:.2f}")
    c4.metric("Avg mean length of run", f"{avg_mlr:.2f}")

    st.plotly_chart(plot_radar_chart(avg_profile_scores), use_container_width=True)

    weakest = min(avg_profile_scores, key=avg_profile_scores.get)
    strongest = max(avg_profile_scores, key=avg_profile_scores.get)

    st.markdown(
        f"""
Your strongest area is **{strongest}**.  
The area that needs the most attention now is **{weakest}**.
"""
    )

    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf_bytes = build_pdf_report(
        user_name=st.session_state.user_name,
        start_time=st.session_state.session_start_time,
        end_time=report_time,
        level_name=level_name,
        avg_score=avg_score,
        avg_profile_scores=avg_profile_scores,
        avg_metrics={
            "speech_rate": avg_speech_rate,
            "articulation_rate": avg_articulation_rate,
            "pause_ratio": avg_pause_ratio,
            "mlr": avg_mlr,
        },
        band_text=fluency_band(avg_score),
    )

    st.download_button(
        label="Generate PDF Report",
        data=pdf_bytes,
        file_name=f"{st.session_state.user_name.replace(' ', '_')}_fluency_report.pdf",
        mime="application/pdf",
    )

    if st.button("Start this level again"):
        reset_level_progress()
