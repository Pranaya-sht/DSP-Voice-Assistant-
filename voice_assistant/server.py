import sys
import os
import time
import json
import traceback
from pathlib import Path

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

TEMP_AUDIO_DIR.mkdir(exist_ok=True, parents=True)
DASHBOARD_STATE_DIR.mkdir(exist_ok=True, parents=True)

# Mount static audio assets for web playback
app.mount("/audio/tts", StaticFiles(directory=str(TEMP_AUDIO_DIR)), name="tts")
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
        ai_response = query_ai(
            user_text=user_text,
            history=history,
            backend="gemini",
            max_tokens=1024
        )
        
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
    """Process an uploaded audio file (or browser microphone WAV recording)."""
    import numpy as np
    
    # Save upload to a temp file inside TEMP_AUDIO_DIR
    temp_file_path = TEMP_AUDIO_DIR / f"upload_{int(time.time()*1000)}.wav"
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

if __name__ == "__main__":
    import uvicorn
    # If run directly, launch uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
