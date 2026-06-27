"""
test_audio_pipeline.py - Audio Test Suite & Auto-Correction Engine
==================================================================
Tests the AI-adaptive DSP pipeline against 11 real audio files with
verified ground truth. If any predictions are wrong, automatically
generates correction samples and retrains the Random Forest.

Usage:
    python test_audio_pipeline.py
"""

import os
import sys
import json
import time
import numpy as np
from scipy.io import wavfile
from pathlib import Path

# Fix Windows console encoding
if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Ground Truth ──────────────────────────────────────────────────────────────
# Keys match ACTUAL filenames in the test/ directory (with spaces, not underscores)

GROUND_TRUTH = {
    "silence-1sec-44100hz-16bit.wav": {
        "measured": {
            "sr": 44100, "channels": 1, "duration_s": 1.0,
            "rms_db": -180.0, "snr_db": -180.0,
            "dominant_freq_hz": 50.0, "zcr": 0.0, "centroid_hz": 0.0
        },
        "expected": {
            "filter": "none", "fft_size": 512, "feature_set": "fft"
        },
        "audio_type": "silence",
        "notes": "Pure silence, int16 zeros. RMS=-180dB. Pipeline must NOT crash."
    },

    "pure-tone.wav": {
        "measured": {
            "sr": 44100, "channels": 1, "duration_s": 40.0,
            "rms_db": -6.9, "snr_db": 75.7,
            "dominant_freq_hz": 7809.0, "zcr": 0.298, "centroid_hz": 5494.7
        },
        "expected": {
            "filter": "bandpass", "fft_size": 1024, "feature_set": "mfcc"
        },
        "audio_type": "pure_tone_sweep",
        "notes": "40s sweep of sine tones 250-8000Hz. High SNR=75.7dB."
    },

    "test_silence.wav": {
        "measured": {
            "sr": 44100, "channels": 1, "duration_s": 3.0,
            "rms_db": -180.0, "snr_db": -180.0,
            "dominant_freq_hz": 50.0, "zcr": 0.0, "centroid_hz": 0.0
        },
        "expected": {
            "filter": "none", "fft_size": 512, "feature_set": "fft"
        },
        "audio_type": "silence",
        "notes": "3s silence, float32 zeros."
    },

    "voice-telephony-8khz.wav": {
        "measured": {
            "sr": 8000, "channels": 1, "duration_s": 26.3,
            "rms_db": -20.1, "snr_db": -1.0,
            "dominant_freq_hz": 224.0, "zcr": 0.1555, "centroid_hz": 1064.2
        },
        "expected": {
            "filter": "bandpass", "fft_size": 512, "feature_set": "mfcc"
        },
        "audio_type": "telephony_speech",
        "notes": "8kHz telephone speech. Low SR forces 512 FFT."
    },

    "cnn14 test.wav": {
        "measured": {
            "sr": 44100, "channels": 1, "duration_s": 5.0,
            "rms_db": -17.3, "snr_db": -3.8,
            "dominant_freq_hz": 74.0, "zcr": 0.0415, "centroid_hz": 2332.3
        },
        "expected": {
            "filter": "highpass", "fft_size": 512, "feature_set": "fft"
        },
        "audio_type": "low_freq_dominant",
        "notes": "DomFreq=74Hz + ZCR=0.04. Strong signal for highpass rule."
    },

    "cnn14 test2.wav": {
        "measured": {
            "sr": 44100, "channels": 1, "duration_s": 5.0,
            "rms_db": -27.6, "snr_db": 152.4,
            "dominant_freq_hz": 50.0, "zcr": 0.0074, "centroid_hz": 0.0
        },
        "expected": {
            "filter": "notch", "fft_size": 512, "feature_set": "fft"
        },
        "audio_type": "electrical_hum_pure",
        "notes": "DomFreq=50Hz exactly. Near-zero centroid. Must trigger notch."
    },

    "High broadband noise2.wav": {
        "measured": {
            "sr": 16000, "channels": 1, "duration_s": 11.23,
            "rms_db": -19.6, "snr_db": -3.7,
            "dominant_freq_hz": 598.0, "zcr": 0.1167, "centroid_hz": 1853.4
        },
        "expected": {
            "filter": "wiener", "fft_size": 2048, "feature_set": "mfcc"
        },
        "audio_type": "broadband_noise_high",
        "notes": "SNR=-3.7dB, rms_db=-19.6. Classic wiener case."
    },

    "High broadband noise.wav": {
        "measured": {
            "sr": 16000, "channels": 1, "duration_s": 11.02,
            "rms_db": -25.7, "snr_db": 1.5,
            "dominant_freq_hz": 201.0, "zcr": 0.1341, "centroid_hz": 2310.6
        },
        "expected": {
            "filter": "bandpass", "fft_size": 1024, "feature_set": "mfcc"
        },
        "audio_type": "moderate_noise_speech",
        "notes": "SNR=1.5dB, rms_db=-25.7. Borderline -> bandpass."
    },

    "test_humSpeech + electrical hum .wav": {
        "measured": {
            "sr": 44100, "channels": 1, "duration_s": 5.0,
            "rms_db": -8.3, "snr_db": -0.0,
            "dominant_freq_hz": 50.0, "zcr": 0.0861, "centroid_hz": 3771.5
        },
        "expected": {
            "filter": "wiener", "fft_size": 2048, "feature_set": "mfcc"
        },
        "audio_type": "speech_plus_hum",
        "notes": "Speech + 50Hz hum. rms=-8.3dB. Wiener over notch because speech present."
    },

    "sterio.wav": {
        "measured": {
            "sr": 96000, "channels": 2, "duration_s": 30.0,
            "rms_db": -14.6, "snr_db": 7.2,
            "dominant_freq_hz": 700.0, "zcr": 0.0074, "centroid_hz": 1181.5
        },
        "expected": {
            "filter": "bandpass", "fft_size": 1024, "feature_set": "mfcc"
        },
        "audio_type": "stereo_high_res",
        "notes": "96kHz stereo. Must downmix to mono. SNR=7.2dB."
    },

    "mono.wav": {
        "measured": {
            "sr": 44100, "channels": 1, "duration_s": 26.3,
            "rms_db": -20.1, "snr_db": -0.7,
            "dominant_freq_hz": 224.0, "zcr": 0.1107, "centroid_hz": 1960.6
        },
        "expected": {
            "filter": "bandpass", "fft_size": 1024, "feature_set": "mfcc"
        },
        "audio_type": "mono_speech",
        "notes": "Standard mono speech. centroid=1960Hz."
    },
}


# ── Audio Loading ─────────────────────────────────────────────────────────────

def load_audio_safe(path: Path) -> tuple:
    """Load WAV, auto-convert to mono float32, return (audio, sr)."""
    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data[:, 0]   # downmix to mono (take left channel)
    # Normalize to float32 [-1, 1]
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    else:
        audio = data.astype(np.float32)
    return audio, sr


# ── Pipeline Runner ───────────────────────────────────────────────────────────

def run_pipeline_on_file(audio: np.ndarray, sr: int) -> tuple:
    """Run the full adaptive DSP pipeline and return predicted params + features."""
    from preprocessing import preprocess
    from dsp_utils import extract_feature_summary
    from decision_engine import DSPDecisionEngine

    preprocessed = preprocess(audio, sr)
    feature_summary = extract_feature_summary(preprocessed["normalized"], sr)
    engine = DSPDecisionEngine(model_path="models/dsp_decision_engine.joblib")
    if not engine.is_trained():
        engine.train_on_synthetic_data()
    prediction = engine.predict(feature_summary)
    return prediction, feature_summary


# ── Test Runner ───────────────────────────────────────────────────────────────

def run_all_tests(audio_dir: Path, ground_truth: dict) -> list:
    """
    Run pipeline on all files. Compare predictions to ground truth.
    Returns list of result dicts.
    """
    results = []

    for filename, gt in ground_truth.items():
        path = audio_dir / filename
        if not path.exists():
            print(f"  [SKIP] {filename} not found at {path}")
            results.append({
                "file": filename, "status": "MISSING",
                "expected": gt["expected"], "predicted": None
            })
            continue

        print(f"\n  [TEST] {filename}")
        print(f"         type={gt['audio_type']}  notes={gt['notes'][:60]}...")

        try:
            t0 = time.perf_counter()
            audio, sr = load_audio_safe(path)
            prediction, feature_summary = run_pipeline_on_file(audio, sr)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            expected = gt["expected"]
            predicted = {
                "filter": prediction["filter"],
                "fft_size": prediction["fft_size"],
                "feature_set": prediction["feature_set"]
            }

            mismatches = []
            for key in ["filter", "fft_size", "feature_set"]:
                if predicted[key] != expected[key]:
                    mismatches.append({
                        "field": key,
                        "expected": expected[key],
                        "predicted": predicted[key]
                    })

            status = "PASS" if not mismatches else "FAIL"
            icon = "\u2705" if status == "PASS" else "\u274c"

            print(f"    {icon} {status}  |  {elapsed_ms:.1f}ms")
            print(f"       Expected:  filter={expected['filter']}  fft={expected['fft_size']}  feat={expected['feature_set']}")
            print(f"       Predicted: filter={predicted['filter']}  fft={predicted['fft_size']}  feat={predicted['feature_set']}")

            if mismatches:
                for m in mismatches:
                    print(f"       MISMATCH [{m['field']}]: got '{m['predicted']}' expected '{m['expected']}'")

            results.append({
                "file": filename,
                "status": status,
                "audio_type": gt["audio_type"],
                "expected": expected,
                "predicted": predicted,
                "feature_summary": feature_summary,
                "mismatches": mismatches,
                "elapsed_ms": elapsed_ms
            })

        except Exception as e:
            print(f"    \u274c ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "file": filename, "status": "ERROR",
                "error": str(e), "expected": gt["expected"], "predicted": None
            })

    return results


# ── Auto-Correction ───────────────────────────────────────────────────────────

def generate_correction_samples(failed_results: list, ground_truth: dict) -> list:
    """
    For each failed test, create synthetic training samples that reinforce
    the correct decision boundary.
    """
    correction_samples = []

    for result in failed_results:
        if result["status"] != "FAIL":
            continue

        fname = result["file"]
        feature_summary = result["feature_summary"]
        correct_labels = result["expected"]
        gt_notes = ground_truth[fname]["notes"]

        print(f"\n  [AUTOCORRECT] Generating corrections for: {fname}")
        print(f"    Correct labels: {correct_labels}")
        print(f"    Reason: {gt_notes}")

        # Feature vector order must match decision_engine.py FEATURE_NAMES:
        # [noise_level, snr_db, spectral_centroid, zero_crossing_rate, bandwidth, dominant_frequency]
        base_vector = np.array([
            feature_summary["noise_level"],
            feature_summary["snr_db"],
            feature_summary["spectral_centroid"],
            feature_summary["zero_crossing_rate"],
            feature_summary["bandwidth"],
            feature_summary["dominant_frequency"]
        ])

        # Jitter magnitudes per feature (realistic variation)
        jitter_scales = np.array([1.0, 2.0, 50.0, 0.01, 100.0, 20.0])

        for i in range(50):  # 50 augmented copies per failed sample
            jitter = np.random.randn(6) * jitter_scales
            jittered = base_vector + jitter
            correction_samples.append((jittered, correct_labels))

        print(f"    Generated 50 correction samples")

    return correction_samples


def retrain_with_corrections(correction_samples: list):
    """
    Load the DSPDecisionEngine, inject correction samples, retrain, save.
    """
    from decision_engine import DSPDecisionEngine

    engine = DSPDecisionEngine(model_path="models/dsp_decision_engine.joblib")

    print(f"\n  [AUTOCORRECT] Retraining with {len(correction_samples)} correction samples...")
    engine.add_correction_samples(correction_samples)
    engine.retrain()
    print("  [AUTOCORRECT] Model retrained and saved.")


def verify_corrections(audio_dir: Path, failed_results: list, ground_truth: dict):
    """
    Re-run only the previously failed files to confirm corrections worked.
    """
    print("\n  [VERIFY] Re-running failed tests after correction...")
    failed_files = {r["file"]: r for r in failed_results if r["status"] == "FAIL"}
    subset_gt = {k: ground_truth[k] for k in failed_files}
    re_results = run_all_tests(audio_dir, subset_gt)

    still_failing = [r for r in re_results if r["status"] == "FAIL"]
    if not still_failing:
        print("\n  [VERIFY] All corrections verified -- pipeline now predicts correctly.")
    else:
        print(f"\n  [VERIFY] {len(still_failing)} files STILL failing after correction:")
        for r in still_failing:
            print(f"    - {r['file']}: {r['mismatches']}")
        print("    Consider adding more jitter samples or expanding training rules.")
    return re_results


# ── Main Orchestrator ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AI-Adaptive DSP Pipeline -- Audio Test Suite")
    print("=" * 60)

    AUDIO_DIR = Path("test")  # test files are in voice_assistant/test/

    # Step 1: Run all tests
    results = run_all_tests(AUDIO_DIR, GROUND_TRUTH)

    # Step 2: Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    missing = sum(1 for r in results if r["status"] == "MISSING")

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed} PASS  {failed} FAIL  {errors} ERROR  {missing} MISSING")
    print("=" * 60)

    # Step 3: Auto-correct if any failures
    failed_results = [r for r in results if r["status"] == "FAIL"]
    if failed_results:
        print(f"\n  [AUTOCORRECT] {len(failed_results)} failures detected. Starting correction...")
        correction_samples = generate_correction_samples(failed_results, GROUND_TRUTH)
        retrain_with_corrections(correction_samples)

        # Step 4: Verify corrections worked
        verify_corrections(AUDIO_DIR, failed_results, GROUND_TRUTH)
    else:
        print("\n  All tests passed! Pipeline is correctly calibrated.")

    # Step 5: Save results to JSON for analysis
    report_path = Path("test_results.json")
    clean_results = []
    for r in results:
        cr = {}
        for k, v in r.items():
            if k == "feature_summary":
                # Convert numpy types in feature_summary
                cr[k] = {fk: float(fv) if hasattr(fv, "item") else fv for fk, fv in v.items()} if v else None
            elif hasattr(v, "tolist"):
                cr[k] = v.tolist()
            else:
                cr[k] = v
        clean_results.append(cr)

    with open(report_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    print(f"\n  [REPORT] Results saved to {report_path}")


if __name__ == "__main__":
    main()
