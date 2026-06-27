import streamlit as st
import json
import time
import numpy as np
import os
import io
import scipy.io.wavfile as wav
import plotly.graph_objects as go
from pathlib import Path

# ── Load environment variables at startup ────────────────────────
def load_env_file():
    env_paths = [Path(".env"), Path(__file__).parent / ".env"]
    for env_path in env_paths:
        if env_path.exists() and env_path.is_file():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip()
                            if val.startswith(('"', "'")) and val.endswith(val[0]):
                                val = val[1:-1]
                            if key and key not in os.environ:
                                os.environ[key] = val
                break
            except Exception:
                pass

load_env_file()

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="DSP Voice Assistant — Interactive Web Workspace",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

STATE_FILE    = Path("dashboard_state/latest_turn.json")
HISTORY_FILE  = Path("dashboard_state/turn_history.json")

# ── Theme Toggle Pattern ───────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # default to dark mode for rich aesthetics

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# ── CSS Design System ──────────────────────────────────────────
BG = "#09090b" if IS_DARK else "#ffffff"
BG_SUBTLE = "#0c0c0f" if IS_DARK else "#f9fafb"
CARD = "#0c0c0f" if IS_DARK else "#ffffff"
CARD_HOVER = "#131316" if IS_DARK else "#f4f4f5"
BORDER = "#1e1e24" if IS_DARK else "#e4e4e7"
BORDER_SUBTLE = "#16161a" if IS_DARK else "#f0f0f2"
TEXT = "#fafafa" if IS_DARK else "#09090b"
TEXT_MUTED = "#71717a"
TEXT_DIM = "#52525b" if IS_DARK else "#a1a1aa"
SHADOW = "none" if IS_DARK else "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"

custom_css = f"""
<style>
    /* Global App Styling */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
        background-color: {BG} !important;
        color: {TEXT} !important;
        font-family: 'DM Sans', -apple-system, sans-serif !important;
    }}
    .block-container {{
        padding: 2rem 2.5rem 3rem !important;
        max-width: 1360px !important;
    }}
    
    /* Horizontal block gaps */
    [data-testid="stHorizontalBlock"] {{
        gap: 1.25rem !important;
    }}
    
    /* Hide Streamlit chrome buttons */
    .stDeployButton, footer, [data-testid="stHeader"] {{
        display: none !important;
    }}

    /* Card Wrapper */
    .chart-wrap {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: {SHADOW};
        margin-bottom: 1rem;
    }}
    .chart-title {{
        font-size: 0.85rem;
        font-weight: 600;
        color: {TEXT};
    }}
    .chart-subtitle {{
        font-size: 0.72rem;
        color: {TEXT_DIM};
        margin-bottom: 0.8rem;
    }}

    /* Metric Cards */
    .metric-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 1.25rem 1.4rem;
        box-shadow: {SHADOW};
        margin-bottom: 0.8rem;
    }}
    .metric-label {{
        font-size: 0.78rem;
        color: {TEXT_MUTED};
        font-weight: 500;
    }}
    .metric-value {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {TEXT};
        letter-spacing: -0.03em;
    }}

    /* HTML tables */
    .data-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }}
    .data-table th {{
        text-align: left;
        padding: 0.6rem 0.8rem;
        color: {TEXT_MUTED};
        font-weight: 500;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        border-bottom: 1px solid {BORDER};
    }}
    .data-table td {{
        padding: 0.65rem 0.8rem;
        color: {TEXT};
        border-bottom: 1px solid {BORDER_SUBTLE};
    }}
    .data-table tr:last-child td {{
        border-bottom: none;
    }}

    /* Status Badges */
    .badge {{
        display: inline-block;
        padding: 2px 9px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 500;
    }}
    .badge-green {{ color: #22c55e; background: rgba(34,197,94,0.12); }}
    .badge-red {{ color: #ef4444; background: rgba(239,68,68,0.12); }}
    .badge-amber {{ color: #f59e0b; background: rgba(245,158,11,0.12); }}
    .badge-blue {{ color: #2563eb; background: rgba(37,99,235,0.12); }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ── Plotly layout config ───────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#71717a" if not IS_DARK else "#a1a1aa", size=11),
    margin=dict(l=40, r=10, t=20, b=30),
    xaxis=dict(
        gridcolor="rgba(0,0,0,0.06)" if not IS_DARK else "rgba(255,255,255,0.06)",
        zerolinecolor="rgba(0,0,0,0.06)" if not IS_DARK else "rgba(255,255,255,0.06)",
        tickfont=dict(size=10, color="#71717a"),
    ),
    yaxis=dict(
        gridcolor="rgba(0,0,0,0.06)" if not IS_DARK else "rgba(255,255,255,0.06)",
        zerolinecolor="rgba(0,0,0,0.06)" if not IS_DARK else "rgba(255,255,255,0.06)",
        tickfont=dict(size=10, color="#71717a"),
    ),
)

FILTER_COLORS = {
    "bandpass": "#3B82F6",
    "highpass": "#22C55E",
    "lowpass":  "#EAB308",
    "notch":    "#EF4444",
    "wiener":   "#A855F7",
    "none":     "#6B7280",
}
FILTER_ICONS = {
    "bandpass": "🔵", "highpass": "🟢", "lowpass": "🟡",
    "notch": "🔴", "wiener": "🟣", "none": "⚪"
}

# ── Load model utilities (cached loaders) ─────────────────────
@st.cache_resource
def get_whisper_model():
    from faster_whisper import WhisperModel
    # CPU loader with fast int8 quantization
    return WhisperModel("base", device="cpu", compute_type="int8")

@st.cache_resource
def get_dsp_engine():
    from decision_engine import DSPDecisionEngine
    engine = DSPDecisionEngine(model_path="models/dsp_decision_engine.joblib")
    if not engine.is_trained():
        engine.train_on_synthetic_data()
    return engine

# ── Passive Monitor loading functions ─────────────────────────
def load_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return None

def load_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except Exception:
        return []

# ── Local TTS response generation helper ───────────────────────
def generate_tts_audio(text: str) -> str:
    import asyncio
    import edge_tts
    from tts_utils import clean_text_for_speech, ensure_temp_dir, TEMP_AUDIO_DIR
    
    clean = clean_text_for_speech(text)
    if not clean:
        return ""
    
    ensure_temp_dir()
    temp_filename = f"tts_{int(time.time()*1000)}.mp3"
    temp_path = os.path.join(TEMP_AUDIO_DIR, temp_filename)
    
    communicate = edge_tts.Communicate(text=clean, voice="en-US-JennyNeural")
    asyncio.run(communicate.save(temp_path))
    return temp_path

# ── Direct Audio processing pipeline ──────────────────────────
def process_interactive_audio(audio_data: np.ndarray, sample_rate: int) -> None:
    from preprocessing import preprocess
    from dsp_utils import extract_feature_summary, compute_fft
    from adaptive_dsp import run_adaptive_dsp
    from cnn14_classifier import classify as cnn14_classify
    from explainability import compute_shap_frequency_importance
    from audio_utils import save_wav
    from ai_utils import query_ai
    
    # Increment turn
    st.session_state.turn_count += 1
    turn = st.session_state.turn_count
    
    status_box = st.status("🔬 Processing audio turn...", expanded=True)
    
    # 1. Preprocessing
    status_box.write("1. Pre-processing audio waveform...")
    preprocessed = preprocess(audio_data, sample_rate)
    audio_normalized = preprocessed["normalized"]
    
    # 2. Features summary
    status_box.write("2. Extracting spectral and acoustic features...")
    feature_summary = extract_feature_summary(audio_normalized, sample_rate)
    
    # 3. Decision predictions
    status_box.write("3. AI Decision Engine selecting filters...")
    dsp_engine = get_dsp_engine()
    dsp_params = dsp_engine.predict(feature_summary)
    
    # 4. Adaptive DSP Filtering
    status_box.write(f"4. Applying adaptive filter: {dsp_params['filter'].upper()}...")
    adaptive_result = run_adaptive_dsp(audio_normalized, sample_rate, dsp_params)
    filtered_audio = adaptive_result["filtered_audio"]
    
    # 5. Environment classification (CNN14)
    status_box.write("5. Classifying audio context (CNN14 Classifier)...")
    cnn14_preds = cnn14_classify(filtered_audio, sample_rate)
    
    # 6. SHAP Explainability
    status_box.write("6. Computing SHAP frequency band importance...")
    try:
        freq_importance = compute_shap_frequency_importance(
            adaptive_result["features"],
            adaptive_result["feature_set"],
            dsp_engine,
            sample_rate,
            dsp_params["fft_size"]
        )
    except Exception as e:
        freq_importance = {"frequencies": [], "importance": []}
    
    # Save the filtered audio to a WAV file to feed to Whisper
    wav_filename = f"turn_{turn:03d}_{int(time.time())}.wav"
    wav_path = save_wav(filtered_audio, sample_rate, filename=wav_filename)
    
    # 7. Transcription (Whisper)
    status_box.write("7. Transcribing speech to text (Whisper Model)...")
    whisper_model = get_whisper_model()
    segments, info = whisper_model.transcribe(
        wav_path,
        language="en",
        beam_size=5,
        vad_filter=True
    )
    user_text = " ".join([segment.text for segment in segments]).strip()
    
    # Check for empty text
    if not user_text:
        status_box.update(label="⚠️ Speech not recognized", state="error")
        st.warning("No speech detected. Please speak clearly into the microphone.")
        st.session_state.turn_count -= 1
        return
        
    # 8. Query AI (Gemini / OpenAI)
    status_box.write("8. Querying Gemini LLM assistant...")
    ai_response = query_ai(
        user_text=user_text,
        history=st.session_state.history,
        backend="gemini",
        max_tokens=1024
    )
    
    # 9. Speak TTS response
    status_box.write("9. Synthesizing AI neural speech response...")
    tts_path = generate_tts_audio(ai_response)
    st.session_state.tts_audio_path = tts_path
    
    # Downsample waveforms for plotting (reduce to 2000 points)
    downsample_factor = max(1, len(audio_normalized) // 2000)
    raw_ds = audio_normalized[::downsample_factor].tolist()
    filt_ds = filtered_audio[::downsample_factor].tolist()
    
    # Compute raw vs filtered FFT magnitudes for plotting
    raw_fft_freq, raw_fft_mag = compute_fft(audio_normalized, sample_rate)
    filt_fft_freq, filt_fft_mag = compute_fft(filtered_audio, sample_rate)
    
    # Limit FFT plotting to 8000 Hz
    max_idx = np.searchsorted(raw_fft_freq, 8000)
    fft_freqs_plot = raw_fft_freq[:max_idx].tolist()
    raw_fft_mag_plot = raw_fft_mag[:max_idx].tolist()
    filt_fft_mag_plot = filt_fft_mag[:max_idx].tolist()
    
    # Compute top SHAP frequencies
    top_shap = []
    if freq_importance.get("frequencies") is not None and len(freq_importance["frequencies"]) > 0:
        imp_arr = np.array(freq_importance["importance"])
        freq_arr = np.array(freq_importance["frequencies"])
        top_indices = np.argsort(imp_arr)[::-1][:5]
        top_shap = [
            {"freq": float(freq_arr[idx]), "importance": float(imp_arr[idx])}
            for idx in top_indices
        ]
        
    # Build state dictionary
    state = {
        "turn": turn,
        "timestamp": time.time(),
        "transcript": user_text,
        "ai_response": ai_response,
        "pipeline": {
            "filter": dsp_params["filter"],
            "fft_size": dsp_params["fft_size"],
            "feature_set": dsp_params["feature_set"],
            "rule": dsp_params.get("rule", "random_forest"),
            "elapsed_ms": 1000.0 # Placeholder elapsed calculation
        },
        "features": {
            "snr_db": float(feature_summary["snr_db"]),
            "noise_level_db": float(feature_summary["noise_level"]),
            "spectral_centroid_hz": float(feature_summary["spectral_centroid"]),
            "zero_crossing_rate": float(feature_summary["zero_crossing_rate"]),
            "bandwidth_hz": float(feature_summary["bandwidth"]),
            "dominant_freq_hz": float(feature_summary["dominant_frequency"])
        },
        "cnn14": [{"label": p["label"], "probability": float(p["probability"])} for p in cnn14_preds[:3]],
        "freq_importance": {
            "frequencies": [float(f) for f in freq_importance.get("frequencies", [])],
            "importance": [float(i) for i in freq_importance.get("importance", [])],
            "top_frequencies": top_shap
        },
        "waveform": {
            "raw": raw_ds,
            "filtered": filt_ds,
            "duration_s": float(len(audio_normalized) / sample_rate)
        },
        "fft": {
            "frequencies": fft_freqs_plot,
            "raw_magnitude": raw_fft_mag_plot,
            "filtered_magnitude": filt_fft_mag_plot
        }
    }
    
    st.session_state.latest_state = state
    
    # Add to history log
    log_entry = {
        "turn": turn,
        "timestamp": state["timestamp"],
        "filter": state["pipeline"]["filter"],
        "fft_size": state["pipeline"]["fft_size"],
        "snr_db": state["features"]["snr_db"],
        "noise_level_db": state["features"]["noise_level_db"],
        "elapsed_ms": state["pipeline"]["elapsed_ms"],
        "cnn14_top": cnn14_preds[0]["label"] if cnn14_preds else "Unknown",
        "transcript": user_text
    }
    st.session_state.history_log.insert(0, log_entry)
    
    status_box.update(label="✅ Audio processed successfully!", state="complete", expanded=False)

# ── Render visualizations ─────────────────────────────────────
def render_dashboard(state: dict, history: list):
    p = state["pipeline"]
    f = state["features"]
    turn = state["turn"]
    icon  = FILTER_ICONS.get(p["filter"], "❓")

    # ── Brand Logo Header & Theme Toggle ──────────────────────
    head_left, head_right = st.columns([8, 1])
    with head_left:
        st.markdown(f"<h2>🎙️ DSP Voice Assistant — Turn {turn}</h2>", unsafe_allow_html=True)
    with head_right:
        theme_label = "☀️ Light" if IS_DARK else "🌙 Dark"
        st.button(theme_label, on_click=toggle_theme, width="stretch" if hasattr(st, "button") else None)

    # ── Top KPI metrics ──────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Filter</div><div class="metric-value">{icon} {p["filter"].upper()}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">FFT Size</div><div class="metric-value">{p["fft_size"]}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Feature Set</div><div class="metric-value">{p["feature_set"].upper()}</div></div>', unsafe_allow_html=True)
    with c4:
        delta_color = "badge-green" if f["snr_db"] > 5 else ("badge-red" if f["snr_db"] < 0 else "badge-amber")
        st.markdown(f'<div class="metric-card"><div class="metric-label">SNR</div><div class="metric-value">{f["snr_db"]:+.1f} dB</div><span class="badge {delta_color}">Estimated</span></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Noise Floor</div><div class="metric-value">{f["noise_level_db"]:.1f} dB</div></div>', unsafe_allow_html=True)
    with c6:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Rule Fired</div><div class="metric-value" style="font-size:1.2rem; word-break:break-all;">{p["rule"]}</div></div>', unsafe_allow_html=True)

    # ── Transcript + AI Response Cards ────────────────────────────
    if state.get("transcript"):
        st.info(f"🗣️ **You said:** {state['transcript']}")
    if state.get("ai_response"):
        st.success(f"🤖 **AI Assistant:** {state['ai_response']}")

    # Play local browser TTS audio if available in session state
    if st.session_state.get("tts_audio_path") and os.path.exists(st.session_state.tts_audio_path):
        st.audio(st.session_state.tts_audio_path, autoplay=True)

    st.divider()

    # ── Row 1: Interactive Waveforms & FFT Spectrums (Plotly) ─────
    col_wave, col_fft = st.columns(2)

    with col_wave:
        st.markdown('<div class="chart-wrap"><div class="chart-title">🌊 Waveform Analysis</div><div class="chart-subtitle">Raw vs. Adaptive filtered output</div>', unsafe_allow_html=True)
        w = state["waveform"]
        raw = np.array(w["raw"])
        filt = np.array(w["filtered"])
        t_axis = np.linspace(0, w["duration_s"], len(raw))
        
        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(x=t_axis, y=raw, name="Raw Signal", line=dict(color="#6B7280" if not IS_DARK else "#4B5563", width=1)))
        fig_wave.add_trace(go.Scatter(x=t_axis, y=filt, name="Filtered Signal", line=dict(color="#3B82F6", width=1.5)))
        fig_wave.update_layout(**PLOT_LAYOUT, height=220, showlegend=True, xaxis_title="Time (s)", yaxis_title="Amplitude")
        st.plotly_chart(fig_wave, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_fft:
        st.markdown('<div class="chart-wrap"><div class="chart-title">📊 FFT Spectrum Analysis</div><div class="chart-subtitle">Spectral energy profile up to 8 kHz</div>', unsafe_allow_html=True)
        fft_data = state["fft"]
        freqs = np.array(fft_data["frequencies"])
        raw_mag  = np.log1p(np.array(fft_data["raw_magnitude"]))
        filt_mag = np.log1p(np.array(fft_data["filtered_magnitude"]))
        
        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(x=freqs, y=raw_mag, name="Raw Magnitude", line=dict(color="#6B7280" if not IS_DARK else "#4B5563", width=1)))
        fig_fft.add_trace(go.Scatter(x=freqs, y=filt_mag, name="Filtered Magnitude", line=dict(color="#EF4444", width=1.5)))
        fig_fft.update_layout(**PLOT_LAYOUT, height=220, showlegend=True, xaxis_title="Frequency (Hz)", yaxis_title="log(1 + magnitude)")
        st.plotly_chart(fig_fft, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 2: CNN14 & SHAP Importance (Plotly) ───────────────────
    col_cnn, col_shap = st.columns(2)

    with col_cnn:
        st.markdown('<div class="chart-wrap"><div class="chart-title">🧠 CNN14 Audio Classification</div><div class="chart-subtitle">Top environmental context probabilities</div>', unsafe_allow_html=True)
        cnn = state.get("cnn14", [])
        if cnn:
            for pred in cnn:
                prob = pred["probability"]
                label = pred["label"]
                st.progress(prob, text=f"{label} ({prob:.1%})")
        else:
            st.caption("No environmental classifications found.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_shap:
        st.markdown('<div class="chart-wrap"><div class="chart-title">🔍 Frequency Band Importance (SHAP)</div><div class="chart-subtitle">Spectrogram bins driving decision engine</div>', unsafe_allow_html=True)
        fi = state.get("freq_importance", {})
        freqs_imp = fi.get("frequencies", [])
        importance = fi.get("importance", [])
        
        if freqs_imp and importance:
            fig_shap = go.Figure(go.Bar(
                x=freqs_imp,
                y=importance,
                marker_color="#8B5CF6",
                name="SHAP Weight"
            ))
            fig_shap.update_layout(**PLOT_LAYOUT, height=200, xaxis_title="Frequency (Hz)", yaxis_title="Importance")
            st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})
            
            top = fi.get("top_frequencies", [])
            if top:
                st.markdown("<p style='font-size:0.75rem; color:#71717a;'>" +
                    "💡 <b>Top bands:</b> " +
                    ", ".join([f"{t['freq']:.0f}Hz" for t in top[:3]]) + "</p>", unsafe_allow_html=True)
        else:
            st.caption("SHAP parameters not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3: Full Feature Details Table ─────────────────────────
    st.subheader("📐 Acoustic Features Details")
    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    with fc1:
        st.metric("Spectral Centroid", f"{f['spectral_centroid_hz']:.0f} Hz")
    with fc2:
        st.metric("Dominant Frequency", f"{f['dominant_freq_hz']:.0f} Hz")
    with fc3:
        st.metric("Acoustic Bandwidth", f"{f['bandwidth_hz']:.0f} Hz")
    with fc4:
        st.metric("Zero Crossing Rate", f"{f['zero_crossing_rate']:.4f}")
    with fc5:
        st.metric("Noise Floor Level", f"{f['noise_level_db']:.1f} dB")

    st.divider()

    # ── Row 4: Turn History Logs ──────────────────────────────────
    st.subheader("📋 Session History")
    if history:
        rows_html = ""
        for h in history[:10]:  # show last 10 turns
            ts_str = pd_format_ts(h["timestamp"])
            rows_html += f"""
            <tr>
                <td><b>Turn {h['turn']}</b></td>
                <td>{ts_str}</td>
                <td><span class="badge badge-blue">{h['filter'].upper()}</span></td>
                <td><code>{h['fft_size']}</code></td>
                <td>{h['snr_db']:+.1f} dB</td>
                <td>{h['noise_level_db']:.1f} dB</td>
                <td>{h['cnn14_top']}</td>
                <td><i>"{h['transcript']}"</i></td>
            </tr>
            """
        st.markdown(f"""
        <table class="data-table">
            <thead>
                <tr>
                    <th>Turn</th>
                    <th>Time</th>
                    <th>Filter</th>
                    <th>FFT Size</th>
                    <th>SNR</th>
                    <th>Noise</th>
                    <th>Context Tag</th>
                    <th>Transcript</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """, unsafe_allow_html=True)
    else:
        st.caption("No turns completed yet.")

def pd_format_ts(ts) -> str:
    try:
        import pandas as pd
        return pd.to_datetime(ts, unit="s").strftime("%H:%M:%S")
    except Exception:
        return time.strftime("%H:%M:%S", time.localtime(ts))

# ── Main Entrypoint ───────────────────────────────────────────
def main():
    # Sidebar config
    st.sidebar.markdown("# ⚙️ Control Panel")
    app_mode = st.sidebar.radio("Operating Mode", ["🎙️ Interactive Assistant", "📈 Passive Monitor"])
    st.sidebar.divider()
    
    st.sidebar.markdown("**Acoustic Filter Key:**")
    for fname, color in FILTER_COLORS.items():
        st.sidebar.markdown(f'<span style="color:{color}">■</span> {fname.upper()}', unsafe_allow_html=True)
        
    if app_mode == "📈 Passive Monitor":
        st.sidebar.caption("Auto-refreshing dashboard reads from local process state files on turn completion.")
        
        # Reloading variables
        state = load_state()
        history = load_history()
        
        if state is None:
            st.info("⏳ Waiting for local assistant turns...")
            st.markdown("""
            ### Start Assistant in Terminal:
            ```bash
            python main.py --backend gemini
            ```
            """)
        else:
            render_dashboard(state, history)
            
        time.sleep(2.0)
        st.rerun()
        
    else:
        # Interactive Assistant Mode
        st.sidebar.caption("Speak directly through your browser or upload WAV/MP3 files.")
        
        # ── Initialize session state ──────────────────────────────
        if "history" not in st.session_state:
            from ai_utils import ConversationHistory
            st.session_state.history = ConversationHistory(max_history_turns=10)
        if "turn_count" not in st.session_state:
            st.session_state.turn_count = 0
        if "latest_state" not in st.session_state:
            st.session_state.latest_state = None
        if "history_log" not in st.session_state:
            st.session_state.history_log = []
        if "last_processed_audio_hash" not in st.session_state:
            st.session_state.last_processed_audio_hash = None
            
        # ── Sidebar interaction panel ─────────────────────────────
        st.sidebar.markdown("### 🎙️ Audio Input Controls")
        
        input_type = st.sidebar.selectbox("Select Audio Input", ["Browser Microphone", "File Upload (WAV/MP3)", "Quick-Load Test Audio File"])
        
        raw_audio = None
        sr = 44100
        audio_hash = None
        
        if input_type == "Browser Microphone":
            # Streamlit 1.37.0+ Browser Microphone input
            audio_file = st.sidebar.audio_input("Record voice message")
            if audio_file is not None:
                audio_bytes = audio_file.getvalue()
                audio_hash = hash(audio_bytes)
                if audio_hash != st.session_state.last_processed_audio_hash:
                    # Load directly from memory bytes
                    try:
                        sr, raw_audio = wav.read(io.BytesIO(audio_bytes))
                        # convert stereo to mono
                        if len(raw_audio.shape) > 1:
                            raw_audio = raw_audio[:, 0]
                    except Exception as e:
                        st.sidebar.error(f"Failed to read mic audio: {e}")
        elif input_type == "File Upload (WAV/MP3)":
            uploaded_file = st.sidebar.file_uploader("Upload WAV/MP3 file", type=["wav", "mp3"])
            if uploaded_file is not None:
                audio_bytes = uploaded_file.getvalue()
                audio_hash = hash(audio_bytes)
                if audio_hash != st.session_state.last_processed_audio_hash:
                    # Save to temp file and load with main.py's load_audio_file
                    from main import load_audio_file
                    temp_upload_path = f"temp_audio/uploaded_{int(time.time())}" + Path(uploaded_file.name).suffix
                    os.makedirs("temp_audio", exist_ok=True)
                    with open(temp_upload_path, "wb") as f:
                        f.write(audio_bytes)
                    try:
                        raw_audio, sr = load_audio_file(temp_upload_path)
                    except Exception as e:
                        st.sidebar.error(f"Failed to parse uploaded audio: {e}")
                    finally:
                        if os.path.exists(temp_upload_path):
                            try: os.remove(temp_upload_path)
                            except: pass
        else:
            test_dir = Path("test")
            if test_dir.exists() and test_dir.is_dir():
                test_files = [f.name for f in test_dir.glob("*.wav")] + [f.name for f in test_dir.glob("*.mp3")]
                test_files.sort()
                selected_test_file = st.sidebar.selectbox("Select Test File", test_files)
                if st.sidebar.button("Run DSP Pipeline on Test File"):
                    test_file_path = test_dir / selected_test_file
                    from main import load_audio_file
                    try:
                        raw_audio, sr = load_audio_file(str(test_file_path))
                        # Use a timestamp-based hash so they can run the same file multiple times
                        audio_hash = hash(str(test_file_path) + str(time.time()))
                    except Exception as e:
                        st.sidebar.error(f"Failed to load test file: {e}")
            else:
                st.sidebar.error("Test directory 'test/' not found.")
                            
        # If new audio detected, process it
        if raw_audio is not None and audio_hash is not None:
            # Process the turn
            process_interactive_audio(raw_audio, sr)
            st.session_state.last_processed_audio_hash = audio_hash
            st.rerun()
            
        # Reset button in sidebar
        if st.sidebar.button("Clear Assistant Conversation"):
            st.session_state.history.clear()
            st.session_state.turn_count = 0
            st.session_state.latest_state = None
            st.session_state.history_log = []
            st.session_state.last_processed_audio_hash = None
            st.session_state.tts_audio_path = None
            st.rerun()

        # Render visualizers
        if st.session_state.latest_state is None:
            st.info("👋 Welcome! Record a voice message or upload an audio file in the sidebar to begin.")
            st.caption("The 8-stage DSP pipeline details and acoustic charts will render here instantly.")
        else:
            render_dashboard(st.session_state.latest_state, st.session_state.history_log)

if __name__ == "__main__":
    main()
