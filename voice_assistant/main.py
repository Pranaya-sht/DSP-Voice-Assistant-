"""
main.py - Voice Assistant Main Application
==========================================
This is the entry point of the voice assistant.
It orchestrates the full pipeline:

  [Microphone] → [WAV File] → [Whisper STT] → [LLM API] → [TTS] → [Speakers]

Run this file to start the assistant:
    python main.py

Press Enter to start recording, speak, then press Enter again to stop.
Say "quit", "exit", or "goodbye" to shut down the assistant.

Architecture note (ESP32 expansion):
  The record_audio() and speak_response() functions are the ONLY places
  that touch hardware (mic/speakers). To adapt for ESP32:
  - Replace record_audio() with a function that receives WAV data from
    an HTTP POST or WebSocket message sent by the ESP32.
  - Replace speak_response() with a function that sends the TTS audio
    back to the ESP32 via HTTP or WebSocket for playback.
  All DSP, STT, and AI logic stays the same!
"""

# ── Standard Library Imports ──────────────────────────────────────────────────
import os           # File paths, environment variables
import sys          # System exit, Python version check
import time         # Timestamps, timing
import threading    # Run visualization in parallel thread
import argparse     # Command-line argument parsing
from pathlib import Path  # Cross-platform path handling

# Fix Windows console encoding for emoji/Unicode support
if os.name == "nt":
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_env_file():
    """
    Check for a .env file and load key-value pairs into os.environ.
    This allows configuring API keys without setting them manually in the shell.
    """
    env_paths = [
        Path(".env"),
        Path(__file__).parent / ".env"
    ]
    for env_path in env_paths:
        if env_path.exists() and env_path.is_file():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        # Ignore comments and empty lines
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip()
                            # Strip quotes if present
                            if val.startswith(('"', "'")) and val.endswith(val[0]):
                                val = val[1:-1]
                            # Only set if it wasn't already set in the environment
                            if key and key not in os.environ:
                                os.environ[key] = val
                print(f"[Main] Loaded environment variables from {env_path}")
                break  # Only load the first one found
            except Exception as e:
                print(f"[Main] Warning: Could not read {env_path}: {e}")

# Load environment variables before any other imports
load_env_file()

# ── Local Module Imports ──────────────────────────────────────────────────────
# These are the modules we built in this project:
from audio_utils import (
    record_audio,
    save_wav,
    print_audio_stats,
    estimate_noise_level,
    list_microphones,
)
from dsp_utils import (
    plot_audio_dashboard,
    compute_fft,
    get_dominant_frequency,
    compute_rms_energy,
    rms_to_db,
    RealTimeVisualizer,
)
from ai_utils import (
    query_ai,
    ConversationHistory,
    check_wake_word,
    handle_special_commands,
    DEFAULT_BACKEND,
)
from tts_utils import speak_response

# ── AI-Adaptive DSP Pipeline Imports ──────────────────────────────────────────
from preprocessing import preprocess
from dsp_utils import extract_feature_summary
from decision_engine import DSPDecisionEngine
from adaptive_dsp import run_adaptive_dsp
from cnn14_classifier import classify as cnn14_classify
from explainability import compute_shap_frequency_importance, plot_explainability_dashboard
from dashboard_writer import write_turn_state



# ── Third-Party: Faster-Whisper ───────────────────────────────────────────────
# faster-whisper is an optimized version of OpenAI's Whisper model.
# It uses CTranslate2 for faster inference (2-4× faster than original Whisper).
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    print("[WARN] faster-whisper not installed.")
    print("       Run: pip install faster-whisper")
    WHISPER_AVAILABLE = False


# ── matplotlib for non-blocking plots ────────────────────────────────────────
import matplotlib
matplotlib.use("TkAgg")   # Must be set before importing pyplot
import matplotlib.pyplot as plt


# ─── Constants ────────────────────────────────────────────────────────────────

APP_NAME    = "🤖 Voice Assistant"
APP_VERSION = "1.0.0"

# Whisper model sizes (larger = more accurate but slower):
# "tiny"   → ~39M params  → fastest,  lowest accuracy
# "base"   → ~74M params  → fast,     decent accuracy   ← good starting point
# "small"  → ~244M params → medium,   good accuracy
# "medium" → ~769M params → slow,     great accuracy
# "large"  → ~1.5B params → slowest,  best accuracy
DEFAULT_WHISPER_MODEL = "base"

# Recording settings
DEFAULT_RECORD_SECONDS = 5      # How many seconds to record per turn
DEFAULT_SAMPLE_RATE    = 44100  # Hz

# Wake word mode: if True, the assistant only responds after hearing wake word
WAKE_WORD_MODE = False          # Set True to require "Hey Assistant" first

# Plot dashboard after each recording?
SHOW_DSP_PLOT = True

# Save dashboard plots to disk?
SAVE_DSP_PLOTS = True
PLOTS_DIR = "plots"

# Noise estimation at startup?
DO_NOISE_ESTIMATION = False  # Set True to calibrate noise floor on startup


# ─── Whisper STT ──────────────────────────────────────────────────────────────

# Global Whisper model instance (loaded once, reused)
# Loading the model takes 1-10 seconds depending on size and hardware.
_whisper_model: "WhisperModel | None" = None


def load_whisper_model(
    model_size: str = DEFAULT_WHISPER_MODEL,
    device: str = "cpu",        # "cpu" or "cuda" (NVIDIA GPU)
    compute_type: str = "int8", # Quantization: "int8" (fast), "float16", "float32"
) -> "WhisperModel":
    """
    Load the Faster-Whisper speech recognition model.

    The model is downloaded automatically from Hugging Face on first use
    and cached in ~/.cache/huggingface/hub/.

    Parameters:
    -----------
    model_size   : str - Whisper model variant (tiny/base/small/medium/large)
    device       : str - "cpu" for CPU inference, "cuda" for NVIDIA GPU
    compute_type : str - Quantization format:
                         "int8"    → fastest, small size, slight quality drop
                         "float16" → fast on GPU, good quality
                         "float32" → slowest, highest quality (no quantization)

    Returns:
    --------
    WhisperModel - The loaded model ready for transcription

    Why global?
    - Loading takes time (~2-10 seconds for "base")
    - We load once at startup and reuse for every recording
    - This is the "singleton pattern" — one shared instance
    """
    global _whisper_model

    if _whisper_model is not None:
        return _whisper_model  # Already loaded — return cached model

    if not WHISPER_AVAILABLE:
        raise RuntimeError(
            "faster-whisper is not installed.\n"
            "Run: pip install faster-whisper"
        )

    print(f"\n[STT] Loading Whisper model: '{model_size}' "
          f"(device={device}, compute_type={compute_type})")
    if "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
        print("[STT]    Note: HF_TOKEN is not set. If you see rate limit warnings or download")
        print("          errors from Hugging Face, you can set a token in your '.env' file.")
    print("[STT]    (First run downloads the model — may take a minute)")

    start = time.time()
    _whisper_model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )
    elapsed = time.time() - start

    print(f"[STT] ✅ Whisper model loaded in {elapsed:.1f}s")
    return _whisper_model


def transcribe_audio(
    wav_filepath: str,
    language: str = "en",           # Language code: "en", "es", "fr", etc.
    beam_size: int = 5,             # Beam search width (higher = more accurate)
    vad_filter: bool = True,        # Voice Activity Detection filter
    vad_threshold: float = 0.5,     # VAD confidence threshold (0-1)
) -> str:
    """
    Transcribe a WAV audio file to text using Faster-Whisper.

    Parameters:
    -----------
    wav_filepath  : str   - Path to the WAV file to transcribe
    language      : str   - Expected language (None = auto-detect)
    beam_size     : int   - Beam search width. Higher = slightly more accurate
                            but slower. 5 is a good balance.
    vad_filter    : bool  - Apply Voice Activity Detection to filter out
                            silence. Reduces hallucinations on quiet audio.
    vad_threshold : float - How confident VAD needs to be that audio is speech.
                            0.5 is standard. Higher = stricter filtering.

    Returns:
    --------
    str - The transcribed text, or empty string if nothing detected

    How Whisper works:
    - The audio is divided into 30-second chunks (Whisper's context window)
    - Each chunk is converted to a log-mel spectrogram (a type of FFT)
    - A transformer model (encoder-decoder) predicts the text tokens
    - Beam search explores multiple possible transcriptions and picks the best
    - The transcription segments are concatenated and returned

    segments is a generator — we iterate it to get each transcribed segment.
    info contains metadata: detected_language, duration, etc.
    """
    model = load_whisper_model()

    if not os.path.exists(wav_filepath):
        print(f"[STT] ❌ WAV file not found: '{wav_filepath}'")
        return ""

    print(f"\n[STT] 🎤 Transcribing: '{wav_filepath}'")
    print(f"[STT]    Language   : {language or 'auto-detect'}")
    print(f"[STT]    Beam size  : {beam_size}")
    print(f"[STT]    VAD filter : {vad_filter}")

    start = time.time()

    try:
        # model.transcribe() returns a generator of segments + info dict
        # segments is lazy — audio is processed as we iterate
        segments, info = model.transcribe(
            wav_filepath,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters={"threshold": vad_threshold},
        )

        # Collect all segment texts into one string
        # Each segment = one sentence or phrase
        full_text = ""
        for segment in segments:
            # segment.text has the transcribed text for this time window
            # segment.start and segment.end are timestamps in seconds
            print(f"[STT]    [{segment.start:.1f}s → {segment.end:.1f}s] {segment.text.strip()}")
            full_text += segment.text + " "

        elapsed = time.time() - start
        full_text = full_text.strip()

        print(f"\n[STT] ✅ Transcription ({elapsed:.2f}s): \"{full_text}\"")
        print(f"[STT]    Detected language: {info.language} "
              f"(confidence: {info.language_probability:.2%})")

        return full_text

    except Exception as e:
        print(f"[STT] ❌ Transcription failed: {e}")
        return ""


# ─── DSP Visualization ────────────────────────────────────────────────────────

def run_dsp_analysis(
    audio_data,
    sample_rate: int,
    turn_number: int,
) -> None:
    """
    Run DSP analysis and display/save the dashboard plot.

    Called after each recording. Shows:
    - Waveform
    - FFT spectrum with dominant frequency
    - Energy meter with dB level
    - Audio statistics panel

    Parameters:
    -----------
    audio_data   : numpy array - The recorded audio
    sample_rate  : int         - Sample rate in Hz
    turn_number  : int         - Which conversation turn this is (for filename)
    """
    # Print quick stats to terminal
    print_audio_stats(audio_data, sample_rate)

    # Compute dominant frequency for terminal output
    freqs, mags = compute_fft(audio_data, sample_rate)
    dom_freq, _ = get_dominant_frequency(freqs, mags)
    rms         = compute_rms_energy(audio_data)
    db          = rms_to_db(rms)

    print(f"\n[DSP] 📊 Key Metrics:")
    print(f"[DSP]    Dominant Frequency : {dom_freq:.1f} Hz")
    print(f"[DSP]    RMS Energy         : {rms:.4f}")
    print(f"[DSP]    Level (dB)         : {db:.1f} dB")
    print(f"[DSP]    Sample Rate        : {sample_rate} Hz")

    if not SHOW_DSP_PLOT:
        return

    # Determine save path
    save_path = None
    if SAVE_DSP_PLOTS:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        save_path = os.path.join(PLOTS_DIR, f"turn_{turn_number:03d}_analysis.png")

    # Create the dashboard
    fig = plot_audio_dashboard(
        audio_data,
        sample_rate,
        title=f"{APP_NAME} — Turn {turn_number} Audio Analysis",
        save_path=save_path,
    )

    # Show non-blocking so the main loop can continue
    # plt.pause() gives the GUI event loop time to render
    plt.pause(0.1)        # Render the figure
    plt.show(block=False) # Don't block — keep running the main loop

    # Close after a short display time (or user can close manually)
    # We use plt.close() after the response is spoken in main loop


# ─── Banner / UI ──────────────────────────────────────────────────────────────

def print_banner() -> None:
    """Print the startup banner."""
    print("╔══════════════════════════════════════════════════╗")
    print("║  🎙️  DSP Voice Assistant  v2.0                  ║")
    print("║                                                  ║")
    print("║  Press Enter  → Start/Stop recording            ║")
    print("║  Type path    → Load audio file                 ║")
    print("║  'quit'       → Exit                            ║")
    print("║                                                  ║")
    print("║  📊 Live Dashboard:                              ║")
    print("║     streamlit run dashboard.py                  ║")
    print("║     Open: http://localhost:8501                  ║")
    print("╚══════════════════════════════════════════════════╝")


def print_separator(char: str = "─", width: int = 65) -> None:
    """Print a separator line."""
    print(char * width)


def print_turn_header(turn: int) -> None:
    """Print the header for a new conversation turn."""
    print(f"\n{'═'*65}")
    print(f"  TURN {turn}  |  {time.strftime('%H:%M:%S')}")
    print(f"{'═'*65}")


# ─── Recording Flow ──────────────────────────────────────────────────────────

def record_with_enter_to_stop(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_duration: int = 30,     # Safety limit: stop after 30s even without Enter
) -> "tuple[np.ndarray, int] | tuple[None, None]":
    """
    Record audio from microphone. Recording starts immediately and stops
    when the user presses Enter (or after max_duration seconds).

    This is more natural than fixed-duration recording:
    - User can speak for as long as needed
    - User controls exactly when to stop

    Implementation:
    - Recording starts in a background thread via sd.InputStream
    - The main thread waits for Enter key press
    - When Enter is pressed, recording stops and data is returned

    Returns:
    --------
    Tuple of (audio_data, sample_rate) or (None, None) on error
    """
    import numpy as np
    import sounddevice as sd

    print("\n[Record] ⏺️  Recording... Press Enter to STOP.")
    print("[Record]    (Speak naturally, then press Enter when done)")

    # Accumulate audio chunks in a list
    # We append each chunk as it arrives, then concatenate at the end
    audio_chunks = []
    recording_done = threading.Event()  # Event flag to signal stop

    def audio_callback(indata, frames, time_info, status):
        """Called by sounddevice each time a new chunk of audio is ready."""
        if status:
            print(f"[Record] Audio status: {status}")
        if not recording_done.is_set():
            # indata is shape (frames, channels) — copy it to avoid overwrite
            audio_chunks.append(indata.copy())

    # Open an InputStream (continuous recording)
    try:
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            callback=audio_callback,
            blocksize=4096,   # Chunk size: 4096 samples per callback
        )

        with stream:
            # Wait for Enter key press (blocking)
            # input() returns when user presses Enter
            input()
            recording_done.set()  # Signal callback to stop accepting chunks

        if not audio_chunks:
            print("[Record] ⚠️  No audio captured. Was the microphone active?")
            return None, None

        # Concatenate all chunks into one array
        # np.concatenate joins arrays along an axis
        import numpy as np
        audio_data = np.concatenate(audio_chunks, axis=0).flatten()
        duration = len(audio_data) / sample_rate

        print(f"[Record] ✅ Captured {duration:.1f}s of audio "
              f"({len(audio_data):,} samples)")

        return audio_data, sample_rate

    except Exception as e:
        print(f"[Record] ❌ Recording error: {e}")
        return None, None


def run_adaptive_pipeline(
    audio: np.ndarray,
    sample_rate: int,
    dsp_engine: DSPDecisionEngine,
    args,
    turn_count: int,
) -> tuple:
    """
    Runs the 8-stage AI-adaptive DSP pipeline.
    Returns (filtered_audio, pipeline_info, feature_summary, cnn14_preds, freq_importance).
    """
    import time
    start_time = time.perf_counter()
    
    print(f"\n[DSP] Running AI-Adaptive DSP Pipeline (Turn {turn_count})...")
    
    # 1-3. Preprocess (normalize, remove DC offset, framing)
    preprocessed = preprocess(audio, sample_rate)
    
    # 4. Feature Summary
    feature_summary = extract_feature_summary(preprocessed["normalized"], sample_rate)
    
    # 5. AI Decision Engine (Random Forest predictions)
    dsp_params = dsp_engine.predict(feature_summary)
    print(f"[DSP] Decision -> Filter: {dsp_params['filter'].upper()} | FFT: {dsp_params['fft_size']} | Features: {dsp_params['feature_set'].upper()}")
    
    # 6. Adaptive DSP (Apply chosen filter and feature extraction)
    adaptive_result = run_adaptive_dsp(preprocessed["normalized"], sample_rate, dsp_params)
    
    # 7-8. CNN14 Classifier & SHAP Explainability Plot (if plots enabled)
    cnn14_preds = []
    freq_importance = {}
    
    if not args.no_plots:
        print("[DSP] Classifying audio environment and generating explainability plot...")
        cnn14_preds = cnn14_classify(adaptive_result["filtered_audio"], sample_rate)
        
        try:
            freq_importance = compute_shap_frequency_importance(
                adaptive_result["features"],
                adaptive_result["feature_set"],
                dsp_engine,
                sample_rate,
                dsp_params["fft_size"]
            )
            plot_explainability_dashboard(
                rf_importance=dsp_engine.get_feature_importances(),
                freq_importance=freq_importance,
                cnn14_predictions=cnn14_preds,
                dsp_params=dsp_params,
                save_path=f"plots/xai_{turn_count:03d}.png"
            )
        except Exception as e:
            print(f"[DSP] Warning: Explainability dashboard generation failed: {e}")
            
    elapsed = (time.perf_counter() - start_time) * 1000
    print(f"[DSP] Pipeline complete in {elapsed:.2f}ms")
    
    pipeline_info = {
        "filter": dsp_params["filter"],
        "fft_size": dsp_params["fft_size"],
        "feature_set": dsp_params["feature_set"],
        "rule": dsp_params.get("rule", "random_forest"),
        "elapsed_ms": elapsed
    }
    
    return adaptive_result["filtered_audio"], pipeline_info, feature_summary, cnn14_preds, freq_importance



def load_audio_file(file_path_str: str) -> "tuple[np.ndarray, int]":
    """
    Load an audio file and return (audio_data, sample_rate).

    Uses pydub (ffmpeg) as the primary decoder — it handles WAV, WebM,
    OGG, MP3, M4A, FLAC, and any other format ffmpeg supports.
    Falls back to scipy.io.wavfile only for .wav files when pydub fails.

    Returns mono int16 numpy array.
    """
    import numpy as np
    from pathlib import Path

    file_path = Path(file_path_str.strip('"\''))
    ext = file_path.suffix.lower()

    # ── Primary path: pydub + ffmpeg (handles every format) ────────────
    _pydub_err = None
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(file_path))
        # Normalise to mono, 16-bit, native sample rate
        audio = audio.set_channels(1).set_sample_width(2)
        sr = audio.frame_rate
        data = np.frombuffer(audio.raw_data, dtype=np.int16)
        print(f"[Audio] Loaded via pydub: {file_path.name}  "
              f"({sr} Hz, {len(data)/sr:.2f}s, {ext})")
        return data, sr
    except Exception as e:
        _pydub_err = e
        print(f"[Audio] pydub failed for {file_path.name}: {e}")

    # ── Fallback: scipy (only works for standard RIFF WAV) ─────────────
    if ext == ".wav":
        try:
            import scipy.io.wavfile as wav
            sr, data = wav.read(str(file_path))
            if len(data.shape) > 1:
                data = data[:, 0]
            print(f"[Audio] Loaded via scipy: {file_path.name}  ({sr} Hz)")
            return data, sr
        except Exception as scipy_err:
            raise RuntimeError(
                f"Cannot read '{file_path.name}': pydub error: {_pydub_err}; "
                f"scipy error: {scipy_err}"
            ) from scipy_err

    raise RuntimeError(
        f"Cannot read '{file_path.name}' ({ext}): pydub/ffmpeg failed and "
        f"scipy only supports .wav.  Original error: {_pydub_err}"
    )


def main(args: argparse.Namespace) -> None:
    """
    Main application loop — runs the voice assistant.

    Pipeline per turn:
    1. Wait for Enter → Record audio
    2. Save as WAV
    3. Run DSP analysis → show plots
    4. Transcribe with Whisper
    5. Check for special commands / wake word
    6. Query AI
    7. Speak response

    Parameters:
    -----------
    args : argparse.Namespace - Command-line arguments (model, backend, etc.)
    """
    print_banner()

    # ── Startup Configuration ──────────────────────────────────────────────
    print(f"[Main] Configuration:")
    print(f"[Main]   AI Backend    : {args.backend.upper()}")
    print(f"[Main]   Whisper Model : {args.whisper_model}")
    print(f"[Main]   Sample Rate   : {args.sample_rate} Hz")
    print(f"[Main]   Wake Word Mode: {'ON' if WAKE_WORD_MODE else 'OFF'}")
    print(f"[Main]   DSP Plots     : {'ON' if SHOW_DSP_PLOT else 'OFF'}")

    # ── List available microphones ─────────────────────────────────────────
    list_microphones()

    # ── Pre-load Whisper model ─────────────────────────────────────────────
    # Load now so first response isn't slow
    print("\n[Main] Pre-loading Whisper STT model...")
    try:
        load_whisper_model(
            model_size=args.whisper_model,
            device=args.device,
            compute_type=args.compute_type,
        )
    except Exception as e:
        print(f"[Main] ❌ Failed to load Whisper: {e}")
        sys.exit(1)

    # ── Optional: Noise estimation ─────────────────────────────────────────
    if DO_NOISE_ESTIMATION or args.calibrate:
        noise_rms = estimate_noise_level(
            sample_rate=args.sample_rate,
            noise_duration=1.5,
        )
        print(f"[Main] Background noise RMS: {noise_rms:.6f}")
    else:
        noise_rms = 0.0

    # ── Initialize Conversation History ───────────────────────────────────
    history = ConversationHistory(
        system_prompt=ai_utils_get_system_prompt(args),
        max_history_turns=args.max_history,
    )
    print(f"\n[Main] Conversation history initialized (max {args.max_history} turns).")

    # ── Initialize AI-Adaptive DSP Decision Engine ────────────────────────
    print("[Main] [DSP] Initializing AI-Adaptive DSP Decision Engine...")
    dsp_engine = DSPDecisionEngine(model_path="models/dsp_decision_engine.joblib")
    if not dsp_engine.is_trained():
        print("[Main] [DSP] Model not found on disk. Training decision engine on synthetic data...")
        dsp_engine.train_on_synthetic_data()


    # ── Wake Word Status ───────────────────────────────────────────────────
    wake_word_active = WAKE_WORD_MODE
    if wake_word_active:
        print("[Main] 🔔 Wake word mode ON. Say 'Hello Assistant' to activate.")
    else:
        print("[Main] 💬 Wake word mode OFF. Ready to chat!")

    # ── Main Conversation Loop ─────────────────────────────────────────────
    turn = 0
    running = True

    print("\n[Main] Press Enter to start your first recording...\n")

    while running:
        try:
            # ────────────────────────────────────────────────────────────────
            # STEP 1: Wait for user to press Enter, then record (or load file)
            # ────────────────────────────────────────────────────────────────
            turn += 1
            print_turn_header(turn)
            print(f"\n[Main] Press Enter to START recording...")
            print(f"[Main] OR type/paste the path to an audio file (WAV/MP3) and press Enter: ")

            # Wait for user input
            try:
                user_input = input().strip()
            except EOFError:
                # EOFError happens if stdin is redirected (e.g., pipe)
                break

            if not user_input:
                # Record audio (stops when user presses Enter again)
                audio_data, sample_rate = record_with_enter_to_stop(
                    sample_rate=args.sample_rate,
                )
            else:
                # Load the specified audio file
                try:
                    audio_data, sample_rate = load_audio_file(user_input)
                    print(f"[Main] Loaded audio file successfully: '{user_input}' ({len(audio_data):,} samples @ {sample_rate} Hz)")
                except Exception as e:
                    print(f"[Main] ❌ Failed to load audio file: {e}")
                    turn -= 1
                    continue

            if audio_data is None:
                print("[Main] Recording failed. Skipping this turn.")
                turn -= 1
                continue

            # ────────────────────────────────────────────────────────────────
            # NEW: Run AI-Adaptive DSP Pipeline
            # ────────────────────────────────────────────────────────────────
            filtered_audio, pipeline_info, feature_summary, cnn14_preds, freq_importance = run_adaptive_pipeline(
                audio_data, sample_rate, dsp_engine, args, turn
            )

            # ────────────────────────────────────────────────────────────────
            # STEP 2: Save recording as WAV (using filtered audio)
            # ────────────────────────────────────────────────────────────────
            wav_filename = f"turn_{turn:03d}_{int(time.time())}.wav"
            wav_path = save_wav(filtered_audio, sample_rate, filename=wav_filename)

            # ────────────────────────────────────────────────────────────────
            # STEP 4 (DSP): Run DSP analysis and plot dashboard (using filtered audio)
            # ────────────────────────────────────────────────────────────────
            if SHOW_DSP_PLOT:
                run_dsp_analysis(filtered_audio, sample_rate, turn)

            # ────────────────────────────────────────────────────────────────
            # STEP 3: Transcribe speech with Faster-Whisper
            # ────────────────────────────────────────────────────────────────
            print("\n[Main] Transcribing speech...")
            user_text = transcribe_audio(
                wav_filepath=wav_path,
                language=args.language,
                beam_size=args.beam_size,
                vad_filter=not args.no_vad,
            )

            # ────────────────────────────────────────────────────────────────
            # STEP 4: Display recognized text
            # ────────────────────────────────────────────────────────────────
            print_separator()
            print(f"  👤 YOU: {user_text if user_text else '(no speech detected)'}")
            print_separator()

            if not user_text.strip():
                print("[Main] ⚠️  No speech detected. Try speaking louder or closer.")
                turn -= 1  # Don't count empty turns
                continue

            # ────────────────────────────────────────────────────────────────
            # Wake Word Check (if mode is enabled)
            # ────────────────────────────────────────────────────────────────
            if wake_word_active:
                if check_wake_word(user_text):
                    print("[Main] 🔔 Wake word detected! Listening for command...")
                    wake_word_active = False  # Deactivate until next silence
                    speak_response(
                        "Yes? How can I help you?",
                        engine=args.tts_engine,
                    )
                    continue
                else:
                    print("[Main] 💤 Waiting for wake word ('Hello Assistant')...")
                    continue

            # ────────────────────────────────────────────────────────────────
            # Special Command Handling (quit, clear history, etc.)
            # ────────────────────────────────────────────────────────────────
            special_response = handle_special_commands(user_text, history)

            if special_response == "__EXIT__":
                print("\n[Main] 👋 Goodbye! Shutting down...")
                speak_response("Goodbye! Have a great day!", engine=args.tts_engine)
                running = False
                break

            elif special_response is not None:
                # A command was handled locally
                print_separator()
                print(f"  🤖 ASSISTANT: {special_response}")
                print_separator()
                speak_response(special_response, engine=args.tts_engine)
                
                # Re-enable wake word if needed
                if wake_word_active is False and WAKE_WORD_MODE:
                    wake_word_active = True
                continue

            # ────────────────────────────────────────────────────────────────
            # STEP 5+6: Send to AI and receive response
            # ────────────────────────────────────────────────────────────────
            print("\n[Main] Sending to AI...")
            try:
                ai_response = query_ai(
                    user_text=user_text,
                    history=history,
                    backend=args.backend,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except Exception as e:
                ai_response = f"Sorry, I had trouble connecting to the AI. Error: {str(e)[:80]}"
                print(f"[Main] ❌ AI query failed: {e}")

            # ────────────────────────────────────────────────────────────────
            # STEP 7: Display AI response in terminal
            # ────────────────────────────────────────────────────────────────
            print_separator()
            print(f"  🤖 ASSISTANT: {ai_response}")
            print_separator()

            # ────────────────────────────────────────────────────────────────
            # STEP 8+9: Convert to speech and play
            # ────────────────────────────────────────────────────────────────
            print("\n[Main] Converting response to speech...")
            try:
                speak_response(
                    ai_response,
                    engine=args.tts_engine,
                    voice=args.tts_voice,
                )
            except Exception as e:
                print(f"[Main] ❌ TTS failed: {e}")
                print(f"[Main]    Response was: {ai_response}")

            # Close the DSP plot window after speaking (clean up)
            if SHOW_DSP_PLOT:
                plt.close("all")

            # Re-enable wake word mode for next turn if needed
            if WAKE_WORD_MODE:
                wake_word_active = True
                print("[Main] 🔔 Wake word mode re-enabled for next turn.")

            # ────────────────────────────────────────────────────────────────
            # NEW: Write Streamlit Dashboard State
            # ────────────────────────────────────────────────────────────────
            if not args.no_plots:
                try:
                    write_turn_state(
                        turn=turn,
                        pipeline_info=pipeline_info,
                        feature_summary=feature_summary,
                        cnn14_predictions=cnn14_preds,
                        freq_importance=freq_importance,
                        raw_audio=audio_data,
                        filtered_audio=filtered_audio,
                        sample_rate=sample_rate,
                        transcript=user_text,
                        ai_response=ai_response
                    )
                except Exception as e:
                    print(f"[Dashboard] Write failed: {e}")

            print(f"\n[Main] Turn {turn} complete. Total history: {history.turn_count} turns.")

        except KeyboardInterrupt:
            # Ctrl+C pressed
            print("\n\n[Main] 🛑 Interrupted by user. Shutting down...")
            running = False
        except Exception as e:
            print(f"\n[Main] ❌ Unexpected error in main loop: {e}")
            print("[Main]    Continuing to next turn...")
            import traceback
            traceback.print_exc()
            turn -= 1  # Don't count failed turns

    # ── Cleanup ───────────────────────────────────────────────────────────
    plt.close("all")
    print("\n[Main] Session summary:")
    print(f"[Main]   Total turns    : {history.turn_count}")
    print(f"[Main]   Plots saved to : '{PLOTS_DIR}/' (if enabled)")
    print(f"[Main]   WAVs saved to  : 'recordings/'")
    print("\n[Main] Goodbye! 👋\n")


def ai_utils_get_system_prompt(args: argparse.Namespace) -> str:
    """Build the system prompt based on command-line args."""
    # Import here to avoid circular imports at module level
    from ai_utils import DEFAULT_SYSTEM_PROMPT
    return DEFAULT_SYSTEM_PROMPT


# ─── Command-Line Arguments ───────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    argparse is Python's built-in argument parser.
    It generates help text automatically when you run:
        python main.py --help

    Examples:
        python main.py --backend gemini --whisper-model small
        python main.py --tts-engine pyttsx3 --no-plots
        python main.py --language fr --sample-rate 16000
    """
    parser = argparse.ArgumentParser(
        prog="voice_assistant",
        description=f"{APP_NAME} — Real-time voice chatbot with DSP visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── AI Backend ────────────────────────────────────────────────────────
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default=DEFAULT_BACKEND,
        choices=["openai", "gemini"],
        help="AI backend to use for language model responses",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="AI response temperature (0=deterministic, 1=creative)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Maximum tokens in AI response",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=10,
        help="Maximum conversation turns to keep in memory",
    )

    # ── Whisper STT ───────────────────────────────────────────────────────
    parser.add_argument(
        "--whisper-model", "-w",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Faster-Whisper model size",
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="en",
        help="Language code for transcription (e.g., 'en', 'es', 'fr'). None=auto-detect",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Whisper beam search size (higher=more accurate but slower)",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Voice Activity Detection filter in Whisper",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for Whisper inference",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        choices=["int8", "float16", "float32"],
        help="Compute type for Whisper (quantization)",
    )

    # ── Audio ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--sample-rate", "-r",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Microphone sample rate in Hz",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run noise calibration at startup",
    )

    # ── TTS ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--tts-engine",
        type=str,
        default="edge_tts",
        choices=["edge_tts", "pyttsx3"],
        help="Text-to-speech engine",
    )
    parser.add_argument(
        "--tts-voice",
        type=str,
        default="en-US-JennyNeural",
        help="TTS voice name (for edge-tts; run 'edge-tts --list-voices' to see options)",
    )

    # ── Visualization ─────────────────────────────────────────────────────
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable DSP visualization plots",
    )

    return parser.parse_args()


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Fix Windows console encoding ──────────────────────────────────────
    # Windows cmd/PowerShell defaults to cp1252, which can't print emoji.
    # Reconfigure stdout/stderr to UTF-8 with 'replace' for unsupported chars.
    if os.name == "nt":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # Verify Python version (3.8+ required for type hints, walrus operator, etc.)
    if sys.version_info < (3, 8):
        print(f"Python 3.8+ required. You have {sys.version}")
        sys.exit(1)

    # Parse arguments
    args = parse_args()

    # Apply --no-plots flag
    if args.no_plots:
        SHOW_DSP_PLOT = False
        SAVE_DSP_PLOTS = False

    # Run the main application
    try:
        # We need numpy for type hints — import check here
        import numpy as np
        main(args)
    except KeyboardInterrupt:
        print("\n\n[Main] Interrupted. Goodbye! 👋")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Main] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
