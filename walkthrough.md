# FastAPI Workspace & Render Deployment Walkthrough

We have successfully migrated the voice assistant dashboard from **Streamlit** to a lightweight **FastAPI** backend with a custom, high-performance **HTML5 / CSS / JavaScript** single-page application.

This change optimizes the assistant for production, resolves local event loop conflicts (`asyncio.run()`), and provides a clean, single-port deployment format that builds on Render out-of-the-box.

---

## 🛠️ Key Architectural Updates

### 1. Root Dependencies Configuration
- **[requirements.txt](file:///c:/Users/prana/Downloads/DSP/requirements.txt)**: Created a root-level `requirements.txt` redirecting to the `voice_assistant` dependencies:
  ```txt
  -r voice_assistant/requirements.txt
  ```
  This resolves the Render build failure (`No such file or directory: 'requirements.txt'`) because Render clones the repository at the root folder level.

- **[voice_assistant/requirements.txt](file:///c:/Users/prana/Downloads/DSP/voice_assistant/requirements.txt)**: Added `fastapi` and `python-multipart` as core dependencies for the web server.

### 2. FastAPI Backend
- **[server.py](file:///c:/Users/prana/Downloads/DSP/voice_assistant/server.py)**: Implemented an ASGI server replacing `dashboard.py`.
  - **Model pre-loading**: Whisper STT, CNN14, and the Random Forest DSP Decision Engine load once on server startup, keeping runtime latency under target metrics.
  - **Async Pipeline Runner**: Modified the audio pipeline runner (`run_pipeline_on_audio`) to be asynchronous, resolving the `RuntimeError: asyncio.run() cannot be called from a running event loop` issue by directly awaiting Microsoft Edge TTS synthesis (`edge-tts`).
  - **API Endpoints**:
    - `GET /`: Serves the HTML5 dashboard workspace.
    - `POST /api/process`: Accepts microphone recordings or uploaded audio.
    - `POST /api/run-test`: Runs the pipeline on selected files from the `test/` suite.
    - `GET /api/state`: Returns the latest turn JSON state.
    - `GET /api/history`: Returns conversation history logs.
    - `GET /api/test-files`: Lists files in `test/`.
    - `POST /api/clear`: Resets active context and logs.

### 3. Glassmorphism Frontend UI
- **[index.html](file:///c:/Users/prana/Downloads/DSP/voice_assistant/templates/index.html)**: Built a responsive, dark-mode zinc interface utilizing:
  - **HTML5 MediaRecorder API**: Captures voice input directly inside the browser and uploads it dynamically.
  - **Plotly.js CDN**: Renders high-performance interactive waveform comparison, FFT spectrum magnitude, and SHAP frequency importance charts.
  - **Autoplay Response**: Plays neural TTS voice responses directly in the browser.
  - **Test Suite Dropdown**: Allows rapid testing by choosing files directly from the UI.

---

## 🚀 How to Run and Deploy

### 1. Run Locally
Start the FastAPI server locally from the root folder:
```bash
.\voice_assistant\venv\Scripts\python -m uvicorn voice_assistant.server:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in any web browser.

### 2. Deploy to Render
When setting up your Web Service on Render:
1. **Repository URL**: `https://github.com/Pranaya-sht/DSP-Voice-Assistant-.git`
2. **Runtime**: `Python`
3. **Build Command**: `pip install -r requirements.txt` (This will install all root dependencies + subdirectory ones)
4. **Start Command**: `python -m uvicorn voice_assistant.server:app --host 0.0.0.0 --port $PORT`
5. **Environment Variables**: Make sure to set `GEMINI_API_KEY` (and `OPENAI_API_KEY` if used) in Render's dashboard environment variables.

---

## 🧪 Verification & API Check
- ✅ FastAPI server successfully loaded and initialized Whisper, CNN14, and DSP Decision Engine models.
- ✅ Exposes API routes on `http://127.0.0.1:8000/`.
- ✅ Verification client call to `/api/test-files` succeeded, returning all test WAVs.
- ✅ Successfully pushed all changes to branch `main` at: [GitHub Repo](https://github.com/Pranaya-sht/DSP-Voice-Assistant-.git).
