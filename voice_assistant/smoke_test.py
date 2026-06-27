"""
smoke_test.py - End-to-End Pipeline Verification
=================================================
This script feeds a synthetic audio signal (with a 50Hz hum and noise)
through the full 8-stage AI-adaptive DSP pipeline to verify all modules work.
"""

import os
import sys
import time
import numpy as np

# Ensure UTF-8 console output on Windows
if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Import all pipeline modules
from preprocessing import preprocess
from dsp_utils import extract_feature_summary
from decision_engine import DSPDecisionEngine
from adaptive_dsp import run_adaptive_dsp
from cnn14_classifier import classify as cnn14_classify
from explainability import compute_shap_frequency_importance, plot_explainability_dashboard

def main():
    print("==================================================")
    print("🔊 AI-ADAPTIVE DSP VOICE ASSISTANT PIPELINE TEST")
    print("==================================================")
    
    # 1. Generate Synthetic Raw Audio (1.5 seconds @ 16kHz)
    # Signal includes: 1000Hz tone (voice harmonic), 50Hz hum, and Gaussian noise
    sample_rate = 16000
    duration = 1.5
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    tone = 0.4 * np.sin(2 * np.pi * 1000 * t)  # speech-like frequency
    hum = 0.25 * np.sin(2 * np.pi * 50 * t)     # 50 Hz power hum
    noise = 0.15 * np.random.normal(0, 1, len(t))
    
    # Combined signal (simulating noisy microphone input)
    raw_audio = (tone + hum + noise) * 32767.0
    raw_audio = raw_audio.astype(np.int16)
    
    print(f"Generated synthetic test signal:")
    print(f"  Duration    : {duration}s")
    print(f"  Sample Rate : {sample_rate} Hz")
    print(f"  Samples     : {len(raw_audio)}")
    print("-" * 50)
    
    # 2. Initialize Decision Engine
    print("[Stage 1] Initializing Decision Engine...")
    model_path = "models/dsp_decision_engine_test.joblib"
    dsp_engine = DSPDecisionEngine(model_path=model_path)
    
    # Train if not trained
    if not dsp_engine.is_trained():
        print("  Training Decision Engine on synthetic data (takes <1s)...")
        dsp_engine.train_on_synthetic_data(n_samples=1000)
    
    # 3. Preprocess Audio
    print("[Stage 2] Running Preprocessing...")
    t_start = time.perf_counter()
    preprocessed = preprocess(raw_audio, sample_rate)
    print(f"  Preprocessed: Mean={np.mean(preprocessed['normalized']):.6f}, Frames shape={preprocessed['frames'].shape}")
    
    # 4. Extract Feature Summary
    print("[Stage 3] Extracting DSP Feature Summary...")
    feature_summary = extract_feature_summary(preprocessed["normalized"], sample_rate)
    print("  Extracted Features:")
    for k, v in feature_summary.items():
        print(f"    - {k:18s}: {v:.4f}")
        
    # 5. AI Decision Prediction
    print("[Stage 4] Querying Random Forest Decision Engine...")
    dsp_params = dsp_engine.predict(feature_summary)
    print(f"  Chosen Settings:")
    print(f"    - Filter     : {dsp_params['filter'].upper()}")
    print(f"    - FFT Size   : {dsp_params['fft_size']}")
    print(f"    - Feature Set: {dsp_params['feature_set'].upper()}")
    
    # 6. Apply Adaptive DSP
    print("[Stage 5] Executing Adaptive DSP Filter & Feature Extraction...")
    dsp_result = run_adaptive_dsp(preprocessed["normalized"], sample_rate, dsp_params)
    print(f"  Features shape extracted: {dsp_result['features'].shape}")
    
    # 7. CNN14 Audio Event Classification
    print("[Stage 6] Running CNN14 Audio Event Classifier...")
    cnn14_predictions = cnn14_classify(dsp_result["filtered_audio"], sample_rate)
    print("  CNN14 Event Probabilities:")
    for p in cnn14_predictions:
        print(f"    - {p['label']:22s}: {p['probability']:.2%}")
        
    # 8. Compute SHAP Frequency Importance
    print("[Stage 7] Computing SHAP Spectral Importance...")
    freq_importance = compute_shap_frequency_importance(
        features=dsp_result["features"],
        feature_set=dsp_result["feature_set"],
        classifier_model=dsp_engine,
        sample_rate=sample_rate,
        fft_size=dsp_params["fft_size"]
    )
    print(f"  SHAP spectral mapping complete. Top frequencies identified:")
    for tf in freq_importance["top_frequencies"][:3]:
        print(f"    - {tf['freq']:.1f} Hz (SHAP Influence={tf['importance']:.4f})")
        
    # 9. Plot explainability dashboard
    print("[Stage 8] Plotting and saving explainability dashboard...")
    save_path = "plots/xai_smoke_test.png"
    plot_explainability_dashboard(
        rf_importance=dsp_engine.get_feature_importances(),
        freq_importance=freq_importance,
        cnn14_predictions=cnn14_predictions,
        dsp_params=dsp_params,
        save_path=save_path
    )
    print(f"  Dashboard successfully saved to '{save_path}'")
    
    # Clean up test model
    if os.path.exists(model_path):
        os.remove(model_path)
        
    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000
    print("-" * 50)
    print(f"⚡ E2E Pipeline completed successfully in {latency_ms:.2f}ms")
    print("==================================================")
    print("🎉 STATUS: ALL TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    main()
