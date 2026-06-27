import json
import time
import numpy as np
from pathlib import Path

DASHBOARD_DIR = Path("dashboard_state/")
DASHBOARD_DIR.mkdir(exist_ok=True)
STATE_FILE = DASHBOARD_DIR / "latest_turn.json"
HISTORY_FILE = DASHBOARD_DIR / "turn_history.json"
MAX_HISTORY = 20   # keep last 20 turns in history

def write_turn_state(
    turn: int,
    pipeline_info: dict,
    feature_summary: dict,
    cnn14_predictions: list,
    freq_importance: dict,
    raw_audio: np.ndarray,
    filtered_audio: np.ndarray,
    sample_rate: int,
    transcript: str = "",
    ai_response: str = ""
):
    """
    Write the full state of one pipeline turn to disk.
    Streamlit dashboard reads this file and re-renders.
    """

    # Downsample audio arrays for display (max 2000 points)
    def downsample(arr, n=2000):
        if len(arr) > n:
            step = len(arr) // n
            return arr[::step][:n].tolist()
        return arr.tolist()

    # Compute FFT for display
    from numpy.fft import rfft, rfftfreq
    N = min(len(raw_audio), sample_rate)
    raw_fft = np.abs(rfft(raw_audio[:N]))
    filt_fft = np.abs(rfft(filtered_audio[:N]))
    freqs = rfftfreq(N, 1/sample_rate).tolist()
    freq_cap = 8000
    cap_idx = next((i for i, f in enumerate(freqs) if f > freq_cap), len(freqs))

    state = {
        "turn": turn,
        "timestamp": time.time(),
        "transcript": transcript,
        "ai_response": ai_response,

        "pipeline": {
            "filter": pipeline_info["filter"],
            "fft_size": pipeline_info["fft_size"],
            "feature_set": pipeline_info["feature_set"],
            "rule": pipeline_info.get("rule", "random_forest"),
            "elapsed_ms": pipeline_info.get("elapsed_ms", 0),
        },

        "features": {
            "snr_db": feature_summary.get("snr_db", 0),
            "noise_level_db": feature_summary.get("noise_level", 0),
            "spectral_centroid_hz": feature_summary.get("spectral_centroid", 0),
            "zero_crossing_rate": feature_summary.get("zero_crossing_rate", 0),
            "bandwidth_hz": feature_summary.get("bandwidth", 0),
            "dominant_freq_hz": feature_summary.get("dominant_frequency", 0),
        },

        "cnn14": cnn14_predictions[:3] if cnn14_predictions else [],

        "freq_importance": {
            "frequencies": freq_importance.get("frequencies", [])[:cap_idx] if freq_importance else [],
            "importance": freq_importance.get("importance", [])[:cap_idx] if freq_importance else [],
            "top_frequencies": freq_importance.get("top_frequencies", []) if freq_importance else []
        },

        "waveform": {
            "raw": downsample(raw_audio),
            "filtered": downsample(filtered_audio),
            "sample_rate": sample_rate,
            "duration_s": len(raw_audio) / sample_rate,
        },

        "fft": {
            "frequencies": freqs[:cap_idx],
            "raw_magnitude": raw_fft[:cap_idx].tolist(),
            "filtered_magnitude": filt_fft[:cap_idx].tolist(),
        }
    }

    # Atomic write (write temp then rename to avoid partial reads)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    
    # Cross-platform atomic rename
    if STATE_FILE.exists():
        STATE_FILE.unlink()
    tmp.rename(STATE_FILE)

    # Append to history
    _append_history(state)
    print(f"[Dashboard] State written for turn {turn}")


def _append_history(state: dict):
    history = []
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE) as f:
                history = json.load(f)
        except Exception:
            history = []

    # Lightweight history entry (no waveform arrays)
    entry = {
        "turn": state["turn"],
        "timestamp": state["timestamp"],
        "transcript": state["transcript"][:80] if state["transcript"] else "—",
        "filter": state["pipeline"]["filter"],
        "fft_size": state["pipeline"]["fft_size"],
        "snr_db": state["features"]["snr_db"],
        "noise_level_db": state["features"]["noise_level_db"],
        "elapsed_ms": state["pipeline"]["elapsed_ms"],
        "cnn14_top": state["cnn14"][0]["label"] if state["cnn14"] else "—",
    }
    history.append(entry)
    history = history[-MAX_HISTORY:]   # keep last 20

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)
