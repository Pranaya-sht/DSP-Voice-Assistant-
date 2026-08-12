
Adaptive DSP Voice Assistant


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference






A real-time voice processing pipeline and interactive web workspace that applies
adaptive Digital Signal Processing (DSP) to audio before performing
Speech-to-Text (STT), Large Language Model (LLM) reasoning, and Text-to-Speech
(TTS).

Instead of sending raw microphone input directly to an LLM, this system analyzes
the signal in real time (SNR, zero-crossing rate, spectral centroid, dominant
frequencies), uses a Random Forest Decision Engine to select and apply acoustic
filters, and feeds both the cleaned audio and DSP metadata into the language
model.

🏗️ Architecture & Pipeline Flow

[ Audio Input ] (Mic / Upload / Test WAV)
       │
       ▼
[ Preprocessing ] (DC Removal, Peak Normalization, Overlapping Framing)
       │
       ▼
[ Feature Summary ] (SNR, Noise Floor, ZCR, Centroid, Bandwidth, Dominant Freq)
       │
       ▼
[ Decision Engine ] (Random Forest + Hardcoded Edge Rules)
       │
       ├──► Predicts: Optimal Filter (Bandpass/Highpass/Lowpass/Notch/Wiener/None),
       │              FFT Window Size (512/1024/2048), Feature Set (MFCC/Wavelet/FFT)
       ▼
[ Adaptive DSP Filtering ] (scipy.signal Butterworth, iirnotch, Wiener)
       │
       ├──► [ CNN14 AudioSet Classifier ] (Environment Context Classification)
       ├──► [ SHAP Spectral Mapping ] (Feature importance back-propagated to frequency bins)
       │
       ▼
[ Speech-to-Text ] (Faster-Whisper on CPU int8)
       │
       ▼
[ LLM Reasoning ] (Google Gemini 2.0 Flash / OpenAI GPT-4o + Injected DSP Context)
       │
       ▼
[ Text-to-Speech ] (Edge-TTS Neural / pyttsx3 Offline)

Key Features

  - Adaptive Audio Cleaning: Dynamically selects Butterworth
    (bandpass/highpass/lowpass), Notch (50/60 Hz hum suppression), or Wiener
    filters based on acoustic metrics.
  - Explainable AI (XAI):
      - SHAP Spectral Mapping: Maps Random Forest feature weights back to
        specific FFT frequency bands driving the filter choice.
      - CNN14 Context Classifier: Identifies ambient sound events (Speech,
        Humming, Room Reverberation, Background Noise) via pretrained AudioSet
        weights.
  - FastAPI + Single-Page Dashboard: Lightweight ASGI web application built with
    a native HTML5/JS interface and interactive Plotly.js charts for waveform,
    spectrum, and SHAP visualization.
  - A/B Audio Comparison: Generates synchronized raw and filtered PCM WAV files
    for side-by-side listening directly in the browser.
  - Context-Aware LLM Reasoning: Prompts the LLM with structured acoustic data
    from the current turn, enabling it to answer technical questions about
    signal quality, noise floor, or filtering choices.
  - Adaptive Explanation Modes: Dynamically changes explanation tone based on
    prompt context or UI selection (Auto, Like I'm 10, College Level,
    Mathematical, Visual, Analogy).
  - Auto-Correction Engine: Includes an automated test suite
    (test_audio_pipeline.py) that evaluates predictions against ground-truth
    audio files and retrains the Random Forest with augmented jitter samples if
    mismatches occur.

📁 Repository Structure

├── Dockerfile                      # Container setup for Hugging Face Spaces (Port 7860)
├── requirements.txt                # Root dependencies redirect (-r voice_assistant/requirements.txt)
├── walkthrough.md                  # Development & migration summary
└── voice_assistant/
    ├── server.py                   # FastAPI ASGI web application & API endpoints
    ├── main.py                     # CLI entry point for local terminal execution
    ├── preprocessing.py            # DC removal, normalization, framing
    ├── dsp_utils.py                # FFT computation, RMS energy, spectral analysis
    ├── decision_engine.py          # Random Forest classifiers + hardcoded rules
    ├── adaptive_dsp.py             # Butterworth, Notch, and Wiener filter chains
    ├── cnn14_classifier.py         # AudioSet classification via PANNs / PyTorch
    ├── explainability.py           # SHAP TreeExplainer & Matplotlib export dashboard
    ├── ai_utils.py                 # OpenAI & Gemini API integration, conversation history
    ├── tts_utils.py                # Edge-TTS & pyttsx3 voice synthesis
    ├── audio_utils.py              # Mic recording, WAV I/O, noise floor estimation
    ├── smoke_test.py               # Pipeline verification with synthetic signal
    ├── test_audio_pipeline.py      # Ground-truth test runner & auto-correction retraining
    ├── models/
    │   └── dsp_decision_engine.joblib  # Trained decision engine model checkpoint
    ├── templates/
    │   └── index.html              # Glassmorphism single-page application UI
    └── test/                       # Audio test suite (.wav files)

⚙️ Installation & Setup

1. Prerequisites

  - Python 3.11+
  - FFmpeg: Required for audio decoding via pydub and edge-tts.
      - Windows: winget install Gyan.FFmpeg
      - macOS: brew install ffmpeg
      - Linux: sudo apt install ffmpeg libportaudio2

2. Environment Setup

Clone the repository and create a virtual environment:

git clone https://github.com/Pranaya-sht/DSP-Voice-Assistant-.git
cd DSP-Voice-Assistant-

python -m venv venv
# Activate on Windows:
.\venv\Scripts\Activate.ps1
# Activate on Linux/macOS:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

3. API Key Configuration

Create a .env file in the voice_assistant/ directory (or use environment
variables):

# AI Models (At least one is required)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional Hugging Face Token (prevents rate limits during Whisper download)
HF_TOKEN=your_hf_token_here

# Runtime Options
AI_BACKEND=gemini          # 'gemini' or 'openai'
FAST_PIPELINE=1            # Set to 1 for faster inference on CPU

🚀 Running the Application

Option A: FastAPI Web Dashboard (Recommended)

Start the Uvicorn server:

python -m uvicorn voice_assistant.server:app --host 127.0.0.1 --port 8000 --reload

Open http://127.0.0.1:8000 in your browser. From the UI, you can:

  - Record voice input via the browser MediaRecorder API.
  - Upload local .wav or .mp3 files.
  - Select and run pipeline tests on preloaded audio files in test/.
  - Toggle "Aggressive Clean" mode or switch explanation styles on the fly.

Option B: Command Line Interface (CLI)

Run the assistant in terminal mode:

cd voice_assistant
python main.py --backend gemini --whisper-model base

CLI options:

  --backend, -b        AI backend: gemini | openai (default: gemini)
  --whisper-model, -w  tiny | base | small | medium | large (default: base)
  --tts-engine         edge_tts | pyttsx3 (default: edge_tts)
  --no-plots           Disable Matplotlib GUI popup

🧪 Testing & Verification

Pipeline Smoke Test

Generates a synthetic 1000 Hz signal mixed with 50 Hz electrical hum and white
noise, then runs all 8 pipeline stages:

python voice_assistant/smoke_test.py

Ground-Truth Test Suite & Auto-Correction

Evaluates the pipeline against 11 test files in voice_assistant/test/. If any
prediction diverges from expected acoustic targets, the auto-correction engine
generates jittered synthetic samples and retrains dsp_decision_engine.joblib:

python voice_assistant/test_audio_pipeline.py

🔌 API Endpoints

The FastAPI server exposes the following routes:

| Method | Endpoint                        | Description                                                                      |
| :----- | :------------------------------ | :------------------------------------------------------------------------------- |
| `GET`  | `/`                             | Serves the HTML5 workspace UI                                                    |
| `POST` | `/api/process`                  | Accepts multipart audio upload (`file`, `explanation_style`, `aggressive_clean`) |
| `POST` | `/api/run-test`                 | Runs the pipeline against a file in `voice_assistant/test/`                      |
| `POST` | `/api/chat`                     | Text follow-up chat using reference DSP context from the previous turn           |
| `GET`  | `/api/state`                    | Returns JSON state for the active turn                                           |
| `GET`  | `/api/history`                  | Returns the session turn log                                                     |
| `GET`  | `/api/audio/turn/{turn}/{kind}` | Streams browser-compatible PCM WAV (`raw` or `filtered`)                         |
| `POST` | `/api/clear`                    | Clears conversation memory and temporary JSON states                             |

🐳 Docker & Hugging Face Spaces Deployment

Build and run locally using Docker:

docker build -t dsp-voice-assistant .
docker run -p 7860:7860 --env-file voice_assistant/.env dsp_voice_assistant

The Dockerfile is pre-configured for Hugging Face Spaces (SDK: Docker,
Port: 7860). All model weights and temporary directories are initialized at
startup with non-root permissions.

