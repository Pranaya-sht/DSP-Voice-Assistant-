import sys
import os
import time
import json
import traceback
import asyncio
from pathlib import Path
import numpy as np

# Add project directories to Python path to ensure clean imports regardless of where server is run
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR.parent))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

def load_env_file():
    """Load key-value pairs from .env into os.environ (if not already set)."""
    env_paths = [Path(".env"), BASE_DIR / ".env", BASE_DIR.parent / ".env"]
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
                            key, val = key.strip(), val.strip()
                            if val.startswith(('"', "'")) and val.endswith(val[0]):
                                val = val[1:-1]
                            if key and key not in os.environ:
                                os.environ[key] = val
                print(f"[Server] Loaded environment from {env_path}")
                break
            except Exception as e:
                print(f"[Server] Warning: Could not read {env_path}: {e}")

load_env_file()

# ── Create FastAPI App ─────────────────────────────────────────
app = FastAPI(
    title="DSP Voice Assistant — Interactive Web API Workspace",
    description="FastAPI + HTML5 web application replacing Streamlit for production deployment."
)

# Ensure folders exist
TEMP_AUDIO_DIR = BASE_DIR / "temp_audio"
DASHBOARD_STATE_DIR = BASE_DIR / "dashboard_state"
RECORDINGS_DIR = BASE_DIR / "recordings"
TEST_DIR = BASE_DIR / "test"
PLOTS_DIR = BASE_DIR / "plots"

TEMP_AUDIO_DIR.mkdir(exist_ok=True, parents=True)
DASHBOARD_STATE_DIR.mkdir(exist_ok=True, parents=True)
RECORDINGS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# Performance tuning (FAST_PIPELINE=1 by default on Spaces CPU)
FAST_PIPELINE = os.environ.get("FAST_PIPELINE", "1").lower() in ("1", "true", "yes")
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "tiny" if FAST_PIPELINE else "base")
WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "1" if FAST_PIPELINE else "5"))
SKIP_CNN14 = os.environ.get("SKIP_CNN14", "1" if FAST_PIPELINE else "0").lower() in ("1", "true", "yes")
VOICE_AI_MAX_TOKENS = int(os.environ.get("VOICE_AI_MAX_TOKENS", "768" if FAST_PIPELINE else "1536"))

# Mount static audio assets for web playback
app.mount("/audio/tts", StaticFiles(directory=str(TEMP_AUDIO_DIR)), name="tts")
app.mount("/audio/recordings", StaticFiles(directory=str(RECORDINGS_DIR)), name="recordings")
app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")
if TEST_DIR.exists():
    app.mount("/audio/test", StaticFiles(directory=str(TEST_DIR)), name="test")

# Global models and state
whisper_model = None
dsp_engine = None
history = None
turn_count = 0
latest_dsp_context: str | None = None
ai_backend = os.environ.get("AI_BACKEND", "gemini")

@app.on_event("startup")
def startup_event():
    """Load Faster-Whisper and DSP Decision Engine models at server startup to ensure instant turn processing."""
    global whisper_model, dsp_engine, history, turn_count
    
    print(f"[Server] Initializing Whisper model ({WHISPER_MODEL_SIZE}, CPU, int8, beam={WHISPER_BEAM_SIZE})...")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    print(f"[Server] Fast pipeline: {FAST_PIPELINE} | Skip CNN14: {SKIP_CNN14}")
    
    print("[Server] Loading DSP Decision Engine...")
    from decision_engine import DSPDecisionEngine
    dsp_engine = DSPDecisionEngine(model_path=str(BASE_DIR / "models" / "dsp_decision_engine.joblib"))
    if not dsp_engine.is_trained():
        print("[Server] DSP engine not trained. Training now...")
        dsp_engine.train_on_synthetic_data()
        
    print("[Server] Initializing Conversation History...")
    from ai_utils import ConversationHistory, build_system_prompt, get_gemini_api_keys, get_openai_api_keys
    history = ConversationHistory(system_prompt=build_system_prompt("auto"))
    gemini_keys = len(get_gemini_api_keys())
    openai_keys = len(get_openai_api_keys())
    print(f"[Server] AI backend: {ai_backend} | Gemini keys: {gemini_keys} | OpenAI keys: {openai_keys}")
    
    # Sync turn count with existing latest turn if it exists
    state_file = DASHBOARD_STATE_DIR / "latest_turn.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                state_data = json.load(f)
                turn_count = int(state_data.get("turn", 0))
                print(f"[Server] Restored turn count to {turn_count} from disk.")
        except Exception:
            pass

# ── Plot Generation Helper ────────────────────────────────────
PLOT_KEY_BY_SUFFIX = {
    "waveform": "waveform_plot",
    "fft_spectrum": "fft_plot",
    "shap": "shap_plot",
    "analysis": "combined_plot",
    "xai": "xai_plot",
}

def get_plot_urls_for_turn(turn: int) -> dict:
    """Reconstruct plot URL map from PNG files saved for a specific turn."""
    prefix = f"turn_{turn:03d}_"
    plot_urls = {}
    if not PLOTS_DIR.exists():
        return plot_urls
    for plot_file in sorted(PLOTS_DIR.glob(f"{prefix}*.png")):
        suffix = plot_file.stem[len(prefix):]
        key = PLOT_KEY_BY_SUFFIX.get(suffix, suffix)
        plot_urls[key] = f"/plots/{plot_file.name}"
    return plot_urls

def persist_turn_state(state: dict) -> None:
    """Write the enriched turn state (including plot_urls) back to disk."""
    state_file = DASHBOARD_STATE_DIR / "latest_turn.json"
    tmp = state_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    if state_file.exists():
        state_file.unlink()
    tmp.rename(state_file)


def _recording_paths(turn: int) -> tuple[Path, Path]:
    raw_path = RECORDINGS_DIR / f"turn_{turn:03d}_raw.wav"
    filtered_path = RECORDINGS_DIR / f"turn_{turn:03d}_filtered.wav"
    return raw_path, filtered_path


def _audio_api_url(turn: int, kind: str) -> str:
    """Use API route for reliable playback on Hugging Face Spaces."""
    return f"/api/audio/turn/{turn}/{kind}"


def save_turn_recordings(
    turn: int,
    raw_audio: np.ndarray,
    filtered_audio: np.ndarray,
    sample_rate: int,
) -> tuple[str, str, str, str]:
    """Save int16 WAVs and return (raw_path, filtered_path, raw_url, filtered_url)."""
    from audio_utils import save_wav

    raw_path, filtered_path = _recording_paths(turn)
    save_wav(raw_audio, sample_rate, filename=str(raw_path))
    save_wav(filtered_audio, sample_rate, filename=str(filtered_path))
    raw_url = _audio_api_url(turn, "raw")
    filtered_url = _audio_api_url(turn, "filtered")
    print(f"[Server] Saved recordings: {filtered_path.name} ({filtered_path.stat().st_size} bytes)")
    return str(raw_path), str(filtered_path), raw_url, filtered_url

def generate_turn_plots(
    turn: int,
    state: dict,
    dsp_engine=None,
    dsp_params: dict | None = None,
    cnn14_preds: list | None = None,
    include_combined: bool = True,
    include_xai: bool = True,
) -> dict:
    """Generate publication-quality matplotlib PNG plots and save to plots/ directory.
    Returns dict of plot URLs for the frontend."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plot_urls = {}
    timestamp = int(time.time())

    # ── Shared style ──────────────────────────────────────────────
    plt.rcParams.update({
        'figure.facecolor': '#0d0d11',
        'axes.facecolor': '#13131a',
        'axes.edgecolor': '#2a2a35',
        'axes.labelcolor': '#a1a1aa',
        'xtick.color': '#71717a',
        'ytick.color': '#71717a',
        'text.color': '#fafafa',
        'grid.color': '#1e1e26',
        'grid.alpha': 0.6,
        'font.family': 'sans-serif',
        'font.size': 10,
    })

    wf = state.get('waveform', {})
    fft_data = state.get('fft', {})
    shap_data = state.get('freq_importance', {})

    # ── 1. Waveform Comparison Plot ───────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
        raw = wf.get('raw', [])
        filtered = wf.get('filtered', [])
        if raw:
            ax.plot(raw, color='#71717a', linewidth=0.6, alpha=0.7, label='Raw Signal')
        if filtered:
            ax.plot(filtered, color='#3b82f6', linewidth=0.8, label='Filtered Signal')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Turn {turn:03d} — Waveform Comparison (Raw vs Filtered)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.3)
        ax.grid(True, axis='y')
        fname = f'turn_{turn:03d}_waveform.png'
        fig.tight_layout()
        fig.savefig(str(PLOTS_DIR / fname), bbox_inches='tight')
        plt.close(fig)
        plot_urls['waveform_plot'] = f'/plots/{fname}'
        print(f'[Plots] Saved waveform: {fname}')
    except Exception as e:
        print(f'[Plots] Waveform plot error: {e}')

    # ── 2. FFT Spectrum Plot ──────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
        freqs = fft_data.get('frequencies', [])
        raw_mag = fft_data.get('raw_magnitude', [])
        filt_mag = fft_data.get('filtered_magnitude', [])
        if freqs and raw_mag:
            ax.plot(freqs, raw_mag, color='#ef4444', linewidth=0.8, alpha=0.7, label='Raw FFT')
        if freqs and filt_mag:
            ax.plot(freqs, filt_mag, color='#10b981', linewidth=0.8, label='Filtered FFT')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title(f'Turn {turn:03d} — FFT Spectrum (up to 8 kHz)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.3)
        ax.grid(True, axis='y')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:.1f}k' if x >= 1000 else f'{x:.0f}'))
        fname = f'turn_{turn:03d}_fft_spectrum.png'
        fig.tight_layout()
        fig.savefig(str(PLOTS_DIR / fname), bbox_inches='tight')
        plt.close(fig)
        plot_urls['fft_plot'] = f'/plots/{fname}'
        print(f'[Plots] Saved FFT spectrum: {fname}')
    except Exception as e:
        print(f'[Plots] FFT plot error: {e}')

    # ── 3. SHAP Frequency Importance Plot ─────────────────────────
    try:
        shap_freqs = shap_data.get('frequencies', [])
        shap_vals = shap_data.get('importance', [])
        if shap_freqs and shap_vals and len(shap_freqs) > 0:
            fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
            colors = ['#10b981' if v >= 0 else '#ef4444' for v in shap_vals]
            ax.bar(range(len(shap_vals)), shap_vals, color=colors, width=1.0, edgecolor='none')
            # Label x-axis with frequency values at intervals
            n_labels = min(10, len(shap_freqs))
            step = max(1, len(shap_freqs) // n_labels)
            tick_pos = list(range(0, len(shap_freqs), step))
            tick_labels = [f'{shap_freqs[i]:.0f}' for i in tick_pos]
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('SHAP Value')
            ax.set_title(f'Turn {turn:03d} — SHAP Frequency Importance', fontsize=12, fontweight='bold')
            ax.axhline(y=0, color='#52525b', linewidth=0.5)
            ax.grid(True, axis='y')
            fname = f'turn_{turn:03d}_shap.png'
            fig.tight_layout()
            fig.savefig(str(PLOTS_DIR / fname), bbox_inches='tight')
            plt.close(fig)
            plot_urls['shap_plot'] = f'/plots/{fname}'
            print(f'[Plots] Saved SHAP: {fname}')
    except Exception as e:
        print(f'[Plots] SHAP plot error: {e}')

    # ── 4. Combined Analysis Plot (all 3 in one) ─────────────────
    if include_combined:
        try:
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=150)
            # Waveform
            raw = wf.get('raw', [])
            filtered = wf.get('filtered', [])
            if raw:
                axes[0].plot(raw, color='#71717a', linewidth=0.5, alpha=0.6, label='Raw')
            if filtered:
                axes[0].plot(filtered, color='#3b82f6', linewidth=0.7, label='Filtered')
            axes[0].set_title('Waveform Comparison', fontsize=11, fontweight='bold')
            axes[0].set_xlabel('Samples')
            axes[0].legend(loc='upper right', framealpha=0.3, fontsize=8)
            axes[0].grid(True, axis='y')
            # FFT
            freqs = fft_data.get('frequencies', [])
            raw_mag = fft_data.get('raw_magnitude', [])
            filt_mag = fft_data.get('filtered_magnitude', [])
            if freqs and raw_mag:
                axes[1].plot(freqs, raw_mag, color='#ef4444', linewidth=0.7, alpha=0.6, label='Raw FFT')
            if freqs and filt_mag:
                axes[1].plot(freqs, filt_mag, color='#10b981', linewidth=0.7, label='Filtered FFT')
            axes[1].set_title('FFT Spectrum', fontsize=11, fontweight='bold')
            axes[1].set_xlabel('Frequency (Hz)')
            axes[1].legend(loc='upper right', framealpha=0.3, fontsize=8)
            axes[1].grid(True, axis='y')
            # SHAP
            shap_freqs = shap_data.get('frequencies', [])
            shap_vals = shap_data.get('importance', [])
            if shap_freqs and shap_vals:
                colors = ['#10b981' if v >= 0 else '#ef4444' for v in shap_vals]
                axes[2].bar(range(len(shap_vals)), shap_vals, color=colors, width=1.0)
                axes[2].set_title('SHAP Frequency Importance', fontsize=11, fontweight='bold')
                axes[2].set_xlabel('Frequency Band')
                axes[2].axhline(y=0, color='#52525b', linewidth=0.5)
            else:
                axes[2].text(0.5, 0.5, 'No SHAP data', ha='center', va='center', color='#71717a', fontsize=12, transform=axes[2].transAxes)
            axes[2].grid(True, axis='y')
            fig.suptitle(f'DSP Pipeline — Turn {turn:03d} Analysis', fontsize=14, fontweight='bold', y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fname = f'turn_{turn:03d}_analysis.png'
            fig.savefig(str(PLOTS_DIR / fname), bbox_inches='tight')
            plt.close(fig)
            plot_urls['combined_plot'] = f'/plots/{fname}'
            print(f'[Plots] Saved combined analysis: {fname}')
        except Exception as e:
            print(f'[Plots] Combined plot error: {e}')

    # ── 5. XAI Explainability Dashboard (RF + SHAP + CNN14 + DSP summary) ──
    if include_xai and dsp_engine is not None and dsp_params is not None:
        try:
            from explainability import plot_explainability_dashboard
            rf_importance = dsp_engine.get_feature_importances()
            fname = f'turn_{turn:03d}_xai.png'
            plot_explainability_dashboard(
                rf_importance=rf_importance,
                freq_importance=state.get("freq_importance", {}),
                cnn14_predictions=cnn14_preds or state.get("cnn14", []),
                dsp_params=dsp_params,
                save_path=str(PLOTS_DIR / fname),
                interactive=False,
            )
            plot_urls["xai_plot"] = f"/plots/{fname}"
            print(f"[Plots] Saved XAI dashboard: {fname}")
        except Exception as e:
            print(f"[Plots] XAI plot error: {e}")

    return plot_urls


async def _generate_plots_background(
    turn: int,
    state: dict,
    dsp_engine,
    dsp_params: dict,
    cnn14_preds: list,
) -> None:
    """Generate PNG plots in a thread so the HTTP response returns faster."""
    loop = asyncio.get_running_loop()
    try:
        plot_urls = await loop.run_in_executor(
            None,
            lambda: generate_turn_plots(
                turn,
                state,
                dsp_engine=dsp_engine,
                dsp_params=dsp_params,
                cnn14_preds=cnn14_preds,
                include_combined=not FAST_PIPELINE,
                include_xai=not FAST_PIPELINE,
            ),
        )
        state["plot_urls"] = plot_urls
        state["plots_ready"] = True
        persist_turn_state(state)
        print(f"[Plots] Background plot generation complete for turn {turn}")
    except Exception as e:
        print(f"[Plots] Background plot generation failed: {e}")

async def synthesize_tts(text: str) -> str | None:
    """Generate TTS audio and return its URL path."""
    from tts_utils import clean_text_for_speech, ensure_temp_dir
    import edge_tts
    clean_text = clean_text_for_speech(text)
    if not clean_text:
        return None
    ensure_temp_dir()
    temp_filename = f"tts_{int(time.time()*1000)}.mp3"
    tts_dest = TEMP_AUDIO_DIR / temp_filename
    communicate = edge_tts.Communicate(text=clean_text, voice="en-US-JennyNeural")
    await communicate.save(str(tts_dest))
    return f"/audio/tts/{temp_filename}"


async def ask_assistant(
    user_text: str,
    dsp_context: str | None = None,
    explanation_style: str = "auto",
    is_followup: bool = False,
    with_tts: bool = False,
    max_tokens: int | None = None,
) -> dict:
    """Query the LLM with optional DSP context and return response payload."""
    global history, latest_dsp_context, ai_backend
    from ai_utils import query_ai, handle_special_commands, style_display_name

    special = handle_special_commands(user_text, history)
    if special:
        tts_url = await synthesize_tts(special) if with_tts else None
        return {
            "ai_response": special,
            "tts_audio_url": tts_url,
            "explanation_style": history.explanation_style,
            "explanation_style_label": style_display_name(history.explanation_style),
            "conversation": history.get_ui_messages(),
            "is_special_command": True,
        }

    ctx = dsp_context if dsp_context is not None else (latest_dsp_context if is_followup else None)
    token_limit = max_tokens if max_tokens is not None else (1536 if is_followup else VOICE_AI_MAX_TOKENS)
    try:
        ai_response = query_ai(
            user_text=user_text,
            history=history,
            backend=ai_backend,
            max_tokens=token_limit,
            dsp_context=ctx,
            explanation_style=explanation_style,
            is_followup=is_followup,
        )
    except Exception as api_err:
        print(f"[Server] AI query failed: {api_err}")
        ai_response = (
            "I'm having trouble reaching the AI service right now — this is often a rate limit or API key issue. "
            "Your message was received. Try again in a moment, or add backup API keys via GEMINI_API_KEYS."
        )

    tts_url = await synthesize_tts(ai_response) if with_tts else None
    return {
        "ai_response": ai_response,
        "tts_audio_url": tts_url,
        "explanation_style": history.explanation_style,
        "explanation_style_label": style_display_name(history.explanation_style),
        "conversation": history.get_ui_messages(),
        "is_special_command": False,
    }

# ── Pipeline Runner Helper ────────────────────────────────────
async def run_pipeline_on_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    explanation_style: str = "auto",
):
    """Run the 8-stage voice assistant and adaptive DSP pipeline asynchronously."""
    global turn_count, history, whisper_model, dsp_engine, latest_dsp_context
    
    from preprocessing import preprocess
    from dsp_utils import extract_feature_summary
    from adaptive_dsp import run_adaptive_dsp
    from cnn14_classifier import classify as cnn14_classify
    from explainability import compute_shap_frequency_importance
    from audio_utils import save_wav
    from ai_utils import build_dsp_context
    from dashboard_writer import write_turn_state
    
    turn_count += 1
    turn = turn_count
    pipeline_start = time.perf_counter()
    
    # 1. Preprocessing
    preprocessed = preprocess(audio_data, sample_rate)
    audio_normalized = preprocessed["normalized"]
    
    # 2. Features summary
    feature_summary = extract_feature_summary(audio_normalized, sample_rate)
    
    # 3. Decision predictions
    dsp_params = dsp_engine.predict(feature_summary)
    
    # 4. Adaptive DSP Filtering
    adaptive_result = run_adaptive_dsp(audio_normalized, sample_rate, dsp_params)
    filtered_audio = adaptive_result["filtered_audio"]
    
    # 5. Environment classification (CNN14) — optional for speed
    if SKIP_CNN14:
        cnn14_preds = []
        print("[Server] CNN14 skipped (SKIP_CNN14=1)")
    else:
        cnn14_preds = cnn14_classify(filtered_audio, sample_rate)
    
    # 6. SHAP Explainability
    try:
        freq_importance = compute_shap_frequency_importance(
            adaptive_result["features"],
            adaptive_result["feature_set"],
            dsp_engine,
            sample_rate,
            dsp_params["fft_size"]
        )
    except Exception as e:
        print(f"[SHAP] Error: {e}")
        freq_importance = {"frequencies": [], "importance": []}

    dsp_params["elapsed_ms"] = (time.perf_counter() - pipeline_start) * 1000

    raw_peak = float(np.max(np.abs(audio_normalized))) if len(audio_normalized) else 0.0
    filt_peak = float(np.max(np.abs(filtered_audio))) if len(filtered_audio) else 0.0
    raw_rms = float(np.sqrt(np.mean(audio_normalized ** 2))) if len(audio_normalized) else 0.0
    filt_rms = float(np.sqrt(np.mean(filtered_audio ** 2))) if len(filtered_audio) else 0.0
    peak_reduction_pct = ((filt_peak - raw_peak) / raw_peak * 100) if raw_peak > 0 else 0.0
    waveform_stats = {
        "duration_s": len(audio_normalized) / sample_rate,
        "sample_rate_hz": sample_rate,
        "raw_peak": raw_peak,
        "filtered_peak": filt_peak,
        "raw_rms": raw_rms,
        "filtered_rms": filt_rms,
        "peak_reduction_pct": peak_reduction_pct,
    }
        
    # Save raw + cleaned audio for playback and Whisper STT
    raw_wav_path, filtered_wav_path, raw_audio_url, filtered_audio_url = save_turn_recordings(
        turn, audio_normalized, filtered_audio, sample_rate
    )
    
    # 7. Transcription (Whisper) — uses noise-cleaned audio
    segments, info = whisper_model.transcribe(
        filtered_wav_path,
        language="en",
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=True
    )
    user_text = " ".join([segment.text for segment in segments]).strip()
    
    # Fallback if Whisper doesn't output anything
    tts_audio_url = None
    if not user_text:
        user_text = "[No speech detected]"
        ai_response = "I couldn't hear or understand anything. Could you please speak again?"
    else:
        # 8. Query AI — only after full DSP analysis + transcription
        dsp_context = build_dsp_context(
            turn=turn,
            pipeline_info=dsp_params,
            feature_summary=feature_summary,
            cnn14_predictions=cnn14_preds,
            freq_importance=freq_importance,
            waveform_stats=waveform_stats,
            rf_importances=dsp_engine.get_feature_importances(),
        )
        latest_dsp_context = dsp_context
        chat_result = await ask_assistant(
            user_text=user_text,
            dsp_context=dsp_context,
            explanation_style=explanation_style,
            is_followup=False,
            with_tts=True,
        )
        ai_response = chat_result["ai_response"]
        tts_audio_url = chat_result["tts_audio_url"]
        
    # 10. Write turn state to disk
    old_cwd = os.getcwd()
    try:
        os.chdir(str(BASE_DIR))
        write_turn_state(
            turn=turn,
            pipeline_info=dsp_params,
            feature_summary=feature_summary,
            cnn14_predictions=cnn14_preds,
            freq_importance=freq_importance,
            raw_audio=audio_normalized,
            filtered_audio=filtered_audio,
            sample_rate=sample_rate,
            transcript=user_text,
            ai_response=ai_response
        )
    finally:
        os.chdir(old_cwd)
        
    # Read the written latest turn state
    state_file = DASHBOARD_STATE_DIR / "latest_turn.json"
    with open(state_file) as f:
        state = json.load(f)
        
    state["tts_audio_url"] = tts_audio_url
    state["raw_audio_url"] = raw_audio_url
    state["filtered_audio_url"] = filtered_audio_url
    state["filter_applied"] = dsp_params.get("filter", "none")
    state["conversation"] = history.get_ui_messages() if history else []
    state["explanation_style"] = history.explanation_style if history else "auto"
    from ai_utils import style_display_name
    state["explanation_style_label"] = style_display_name(state["explanation_style"])
    state["plot_urls"] = {}
    state["plots_ready"] = False
    persist_turn_state(state)

    # Generate PNG plots in background so chat/audio return faster
    asyncio.create_task(
        _generate_plots_background(turn, state, dsp_engine, dsp_params, cnn14_preds)
    )
    
    return state

# ── API Endpoints ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def get_index_page():
    """Serve the main single-page web assistant interface."""
    index_path = BASE_DIR / "templates" / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html template not found.")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/state")
def get_latest_state():
    """Retrieve the latest DSP and pipeline status json."""
    state_file = DASHBOARD_STATE_DIR / "latest_turn.json"
    if not state_file.exists():
        return JSONResponse(content={})
    with open(state_file) as f:
        state = json.load(f)
    if not state.get("plot_urls") and state.get("turn"):
        state["plot_urls"] = get_plot_urls_for_turn(int(state["turn"]))
    turn_num = state.get("turn")
    if turn_num:
        _, filtered_path = _recording_paths(int(turn_num))
        if filtered_path.exists():
            state["filtered_audio_url"] = state.get("filtered_audio_url") or _audio_api_url(int(turn_num), "filtered")
            state["raw_audio_url"] = state.get("raw_audio_url") or _audio_api_url(int(turn_num), "raw")
    if history:
        from ai_utils import style_display_name
        state["conversation"] = history.get_ui_messages()
        state["explanation_style"] = history.explanation_style
        state["explanation_style_label"] = style_display_name(history.explanation_style)
        state["has_dsp_context"] = latest_dsp_context is not None
    return JSONResponse(content=state)


@app.get("/api/audio/turn/{turn}/{kind}")
def get_turn_audio(turn: int, kind: str):
    """Stream saved raw or filtered WAV for a turn (browser-safe int16 PCM)."""
    if kind not in ("raw", "filtered"):
        raise HTTPException(status_code=400, detail="kind must be 'raw' or 'filtered'")
    raw_path, filtered_path = _recording_paths(turn)
    path = raw_path if kind == "raw" else filtered_path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Audio not found for turn {turn} ({kind})")
    return FileResponse(
        path,
        media_type="audio/wav",
        filename=path.name,
        headers={"Accept-Ranges": "bytes"},
    )


@app.get("/api/history")
def get_history_log():
    """Retrieve the conversation turns logs history."""
    history_file = DASHBOARD_STATE_DIR / "turn_history.json"
    if not history_file.exists():
        return JSONResponse(content=[])
    with open(history_file) as f:
        return JSONResponse(content=json.load(f))

@app.get("/api/test-files")
def get_test_files():
    """Get list of files in test directory for the dropdown selection."""
    if not TEST_DIR.exists():
        return JSONResponse(content=[])
    files = [f.name for f in TEST_DIR.glob("*.wav")] + [f.name for f in TEST_DIR.glob("*.mp3")]
    files.sort()
    return JSONResponse(content=files)

@app.post("/api/process")
async def process_audio(
    file: UploadFile = File(...),
    explanation_style: str = Form("auto"),
):
    """Process an uploaded audio file (or browser microphone recording)."""
    import numpy as np
    from pathlib import Path
    
    # Preserve the original extension so pydub/ffmpeg can detect the format
    original_ext = Path(file.filename).suffix.lower() if file.filename else ".wav"
    if not original_ext:
        original_ext = ".wav"
    temp_file_path = TEMP_AUDIO_DIR / f"upload_{int(time.time()*1000)}{original_ext}"
    try:
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())
            
        from main import load_audio_file
        raw_audio, sr = load_audio_file(str(temp_file_path))
        
        # Run processing pipeline
        state = await run_pipeline_on_audio(raw_audio, sr, explanation_style=explanation_style)
        return JSONResponse(content=state)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()

@app.post("/api/run-test")
async def run_test_file(
    filename: str = Form(...),
    explanation_style: str = Form("auto"),
):
    """Process a pre-loaded test audio file from the dropdown selector."""
    import numpy as np
    
    test_file_path = TEST_DIR / filename
    if not test_file_path.exists():
        raise HTTPException(status_code=404, detail="Selected test file does not exist.")
        
    try:
        from main import load_audio_file
        raw_audio, sr = load_audio_file(str(test_file_path))
        
        # Run processing pipeline
        state = await run_pipeline_on_audio(raw_audio, sr, explanation_style=explanation_style)
        return JSONResponse(content=state)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversation")
def get_conversation():
    """Return the live LLM conversation thread."""
    from ai_utils import style_display_name, EXPLANATION_STYLES
    if not history:
        return JSONResponse(content={"messages": [], "explanation_style": "auto"})
    return JSONResponse(content={
        "messages": history.get_ui_messages(),
        "explanation_style": history.explanation_style,
        "explanation_style_label": style_display_name(history.explanation_style),
        "available_styles": list(EXPLANATION_STYLES.keys()),
        "has_dsp_context": latest_dsp_context is not None,
    })


@app.post("/api/chat")
async def chat_followup(
    message: str = Form(...),
    explanation_style: str = Form("auto"),
    with_tts: str = Form("false"),
):
    """Text follow-up chat using the last audio turn's DSP analysis context."""
    global history
    if not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if not history:
        raise HTTPException(status_code=503, detail="Assistant not initialized.")

    tts_enabled = with_tts.lower() in ("true", "1", "yes")
    result = await ask_assistant(
        user_text=message.strip(),
        dsp_context=None,
        explanation_style=explanation_style,
        is_followup=True,
        with_tts=tts_enabled,
    )
    return JSONResponse(content=result)


@app.post("/api/set-style")
def set_explanation_style(style: str = Form(...)):
    """Update the active explanation style without sending a message."""
    from ai_utils import style_display_name, normalize_explanation_style
    if not history:
        raise HTTPException(status_code=503, detail="Assistant not initialized.")
    normalized = history.set_explanation_style(style)
    return JSONResponse(content={
        "explanation_style": normalized,
        "explanation_style_label": style_display_name(normalized),
    })


@app.post("/api/clear")
def clear_workspace():
    """Clear conversation history memory and disk JSON states."""
    global history, turn_count, latest_dsp_context
    if history:
        history.clear()
        history.set_explanation_style("auto")
    turn_count = 0
    latest_dsp_context = None
    
    state_file = DASHBOARD_STATE_DIR / "latest_turn.json"
    if state_file.exists():
        state_file.unlink()
        
    history_file = DASHBOARD_STATE_DIR / "turn_history.json"
    if history_file.exists():
        history_file.unlink()

    if PLOTS_DIR.exists():
        for plot_file in PLOTS_DIR.glob("*.png"):
            plot_file.unlink()
        
    return JSONResponse(content={"status": "cleared"})

@app.get("/api/plots")
def list_plots(turn: int | None = None):
    """List saved plot images in plots/, optionally filtered to one turn."""
    if not PLOTS_DIR.exists():
        return JSONResponse(content=[])
    pattern = f"turn_{turn:03d}_*.png" if turn is not None else "*.png"
    plots = sorted([
        {"name": f.name, "url": f"/plots/{f.name}", "size_kb": round(f.stat().st_size / 1024, 1)}
        for f in PLOTS_DIR.glob(pattern)
    ], key=lambda x: x["name"])
    return JSONResponse(content=plots)

if __name__ == "__main__":
    import uvicorn
    import os
    # Hugging Face Spaces uses PORT=7860 by default
    port = int(os.environ.get("PORT", 7860))
    # If run directly, launch uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
