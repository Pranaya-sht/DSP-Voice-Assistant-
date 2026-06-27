import sys
import os
import time
import json
import traceback
from pathlib import Path
import numpy as np

# Add project directories to Python path to ensure clean imports regardless of where server is run
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR.parent))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Load environment variables
from dashboard import load_env_file
load_env_file()

# ── Create FastAPI App ─────────────────────────────────────────
app = FastAPI(
    title="DSP Voice Assistant — Interactive Web API Workspace",
    description="FastAPI + HTML5 web application replacing Streamlit for production deployment."
)

# Ensure folders exist
TEMP_AUDIO_DIR = BASE_DIR / "temp_audio"
DASHBOARD_STATE_DIR = BASE_DIR / "dashboard_state"
TEST_DIR = BASE_DIR / "test"
PLOTS_DIR = BASE_DIR / "plots"

TEMP_AUDIO_DIR.mkdir(exist_ok=True, parents=True)
DASHBOARD_STATE_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# Mount static audio assets for web playback
app.mount("/audio/tts", StaticFiles(directory=str(TEMP_AUDIO_DIR)), name="tts")
app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")
if TEST_DIR.exists():
    app.mount("/audio/test", StaticFiles(directory=str(TEST_DIR)), name="test")

# Global models and state
whisper_model = None
dsp_engine = None
history = None
turn_count = 0

@app.on_event("startup")
def startup_event():
    """Load Faster-Whisper and DSP Decision Engine models at server startup to ensure instant turn processing."""
    global whisper_model, dsp_engine, history, turn_count
    
    print("[Server] Initializing Whisper model (base, CPU, int8)...")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    
    print("[Server] Loading DSP Decision Engine...")
    from decision_engine import DSPDecisionEngine
    dsp_engine = DSPDecisionEngine(model_path=str(BASE_DIR / "models" / "dsp_decision_engine.joblib"))
    if not dsp_engine.is_trained():
        print("[Server] DSP engine not trained. Training now...")
        dsp_engine.train_on_synthetic_data()
        
    print("[Server] Initializing Conversation History...")
    from ai_utils import ConversationHistory, DEFAULT_SYSTEM_PROMPT
    history = ConversationHistory(system_prompt=DEFAULT_SYSTEM_PROMPT)
    
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
def generate_turn_plots(turn: int, state: dict) -> dict:
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

    return plot_urls

# ── Pipeline Runner Helper ────────────────────────────────────
async def run_pipeline_on_audio(audio_data: np.ndarray, sample_rate: int):
    """Run the 8-stage voice assistant and adaptive DSP pipeline asynchronously."""
    global turn_count, history, whisper_model, dsp_engine
    
    from preprocessing import preprocess
    from dsp_utils import extract_feature_summary
    from adaptive_dsp import run_adaptive_dsp
    from cnn14_classifier import classify as cnn14_classify
    from explainability import compute_shap_frequency_importance
    from audio_utils import save_wav
    from ai_utils import query_ai
    from tts_utils import clean_text_for_speech, ensure_temp_dir
    import edge_tts
    from dashboard_writer import write_turn_state
    
    turn_count += 1
    turn = turn_count
    
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
    
    # 5. Environment classification (CNN14)
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
        
    # Save the filtered audio to a WAV file to feed to Whisper STT
    wav_filename = f"turn_{turn:03d}_{int(time.time())}.wav"
    recordings_dir = BASE_DIR / "recordings"
    recordings_dir.mkdir(exist_ok=True)
    wav_path = save_wav(filtered_audio, sample_rate, filename=str(recordings_dir / wav_filename))
    
    # 7. Transcription (Whisper)
    segments, info = whisper_model.transcribe(
        wav_path,
        language="en",
        beam_size=5,
        vad_filter=True
    )
    user_text = " ".join([segment.text for segment in segments]).strip()
    
    # Fallback if Whisper doesn't output anything
    if not user_text:
        user_text = "[No speech detected]"
        ai_response = "I couldn't hear or understand anything. Could you please speak again?"
    else:
        # 8. Query AI (Gemini)
        try:
            ai_response = query_ai(
                user_text=user_text,
                history=history,
                backend="gemini",
                max_tokens=1024
            )
        except Exception as api_err:
            print(f"[Server] ⚠️ Gemini API Query failed: {api_err}")
            ai_response = f"[API Error / Rate Limit Exceeded] I processed your speech: '{user_text}', but my AI brain was unable to answer because of a Gemini API quota limit. Please check your Gemini key or try again in a few minutes!"
        
    # 9. Speak TTS response (Async edge_tts communicate)
    clean_text = clean_text_for_speech(ai_response)
    if clean_text:
        ensure_temp_dir()
        temp_filename = f"tts_{int(time.time()*1000)}.mp3"
        tts_dest = TEMP_AUDIO_DIR / temp_filename
        
        communicate = edge_tts.Communicate(text=clean_text, voice="en-US-JennyNeural")
        await communicate.save(str(tts_dest))
        tts_audio_url = f"/audio/tts/{temp_filename}"
    else:
        tts_audio_url = None
        
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
    
    # Generate and save matplotlib plots to plots/ directory
    try:
        plot_urls = generate_turn_plots(turn, state)
        state["plot_urls"] = plot_urls
    except Exception as e:
        print(f"[Plots] Error generating plots: {e}")
        state["plot_urls"] = {}
    
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
        return JSONResponse(content=json.load(f))

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
async def process_audio(file: UploadFile = File(...)):
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
        state = await run_pipeline_on_audio(raw_audio, sr)
        return JSONResponse(content=state)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()

@app.post("/api/run-test")
async def run_test_file(filename: str = Form(...)):
    """Process a pre-loaded test audio file from the dropdown selector."""
    import numpy as np
    
    test_file_path = TEST_DIR / filename
    if not test_file_path.exists():
        raise HTTPException(status_code=404, detail="Selected test file does not exist.")
        
    try:
        from main import load_audio_file
        raw_audio, sr = load_audio_file(str(test_file_path))
        
        # Run processing pipeline
        state = await run_pipeline_on_audio(raw_audio, sr)
        return JSONResponse(content=state)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear")
def clear_workspace():
    """Clear conversation history memory and disk JSON states."""
    global history, turn_count
    if history:
        history.clear()
    turn_count = 0
    
    state_file = DASHBOARD_STATE_DIR / "latest_turn.json"
    if state_file.exists():
        state_file.unlink()
        
    history_file = DASHBOARD_STATE_DIR / "turn_history.json"
    if history_file.exists():
        history_file.unlink()
        
    return JSONResponse(content={"status": "cleared"})

@app.get("/api/plots")
def list_plots():
    """List all saved plot images in the plots/ directory."""
    if not PLOTS_DIR.exists():
        return JSONResponse(content=[])
    plots = sorted([
        {"name": f.name, "url": f"/plots/{f.name}", "size_kb": round(f.stat().st_size / 1024, 1)}
        for f in PLOTS_DIR.glob("*.png")
    ], key=lambda x: x["name"], reverse=True)
    return JSONResponse(content=plots)

if __name__ == "__main__":
    import uvicorn
    import os
    # Hugging Face Spaces uses PORT=7860 by default
    port = int(os.environ.get("PORT", 7860))
    # If run directly, launch uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
