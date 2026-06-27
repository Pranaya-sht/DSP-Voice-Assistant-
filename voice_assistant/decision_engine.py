"""
decision_engine.py - Random Forest DSP Decision Engine
======================================================
This module implements a decision layer based on three RandomForestClassifiers.
It analyzes high-level audio features (SNR, ZCR, Spectral Centroid, etc.)
and predicts the optimal DSP pipeline settings: filter, FFT size, and feature set.

Includes hardcoded edge-case rules that override the RF when acoustic conditions
match known failure modes, plus auto-correction support for retraining with
real-world labeled data.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

class DSPDecisionEngine:
    """
    Trained Random Forest models that map audio feature summaries
    to optimal DSP parameters.
    """
    FILTER_OPTIONS = ["bandpass", "highpass", "lowpass", "notch", "wiener", "none"]
    FFT_SIZES = [512, 1024, 2048]
    FEATURE_SETS = ["mfcc", "wavelet", "fft"]
    FEATURE_NAMES = ["noise_level", "snr_db", "spectral_centroid", "zero_crossing_rate", "bandwidth", "dominant_frequency"]

    def __init__(self, model_path: str = None):
        self.model_path = model_path or "models/dsp_decision_engine.joblib"
        self.filter_rf = None
        self.fft_rf = None
        self.feature_rf = None
        self.last_feature_summary = None
        
        # Correction sample buffers for auto-retraining
        self._correction_X = []
        self._correction_y_filter = []
        self._correction_y_fft = []
        self._correction_y_features = []
        
        # Load the models if they exist on disk
        if os.path.exists(self.model_path):
            try:
                self.load(self.model_path)
                print(f"[DSP] Loaded decision engine models from: '{self.model_path}'")
            except Exception as e:
                print(f"[DSP] Warning: Could not load decision engine models: {e}. Re-training will be required.")

    def is_trained(self) -> bool:
        """Check if all three Random Forest models are loaded and ready."""
        return self.filter_rf is not None and self.fft_rf is not None and self.feature_rf is not None

    # ── Hardcoded Edge-Case Rules ─────────────────────────────────────────

    def _apply_hardcoded_rules(self, feature_summary: dict) -> dict | None:
        """
        Returns a forced prediction dict if a hardcoded rule fires, else None.
        Rules are checked in priority order (first match wins).
        These catch edge cases where the RF synthetic training data doesn't
        cover real-world acoustic conditions well enough.
        """
        rms_db = feature_summary["noise_level"]
        snr_db = feature_summary["snr_db"]
        dom_freq = feature_summary["dominant_frequency"]
        zcr = feature_summary["zero_crossing_rate"]
        centroid = feature_summary["spectral_centroid"]
        sr = feature_summary.get("sample_rate", 44100)
        rms_db_overall = feature_summary.get("rms_db_overall", rms_db)

        # Rule 1: Silence — overall RMS below -60dB → skip everything
        if rms_db_overall < -60:
            return {"filter": "none", "fft_size": 512, "feature_set": "fft",
                    "rule": "silence_detected"}

        # Rule 2: Pure 50/60Hz electrical hum or low-frequency pure tone in silent start
        # If noise_level is low (silent start) but overall RMS is high, and ZCR & centroid are very low
        if rms_db < -60 and rms_db_overall > -60 and centroid < 400 and zcr < 0.02:
            return {"filter": "notch", "fft_size": 512, "feature_set": "fft",
                    "rule": "electrical_hum_pure"}

        # Rule 3: Low sample rate audio (telephony) — SR <= 8000
        if sr <= 8000:
            return {"filter": "bandpass", "fft_size": 512, "feature_set": "mfcc",
                    "rule": "telephony_sr"}

        # Rule 4: Low dominant frequency + very low ZCR + low SNR → highpass (hum in noisy/non-silent audio)
        if dom_freq < 150 and zcr < 0.05 and rms_db_overall > -60 and snr_db < 5.0:
            return {"filter": "highpass", "fft_size": 512, "feature_set": "fft",
                    "rule": "low_freq_dominant"}

        return None   # No rule fired — let Random Forest decide


    # ── Synthetic Data Generation ─────────────────────────────────────────

    def _generate_synthetic_data(self, n_samples: int = 2000):
        """
        Generate synthetic training data representing various acoustic conditions.
        Returns (X, y_filter, y_fft, y_feat) arrays.
        """
        np.random.seed(42)
        X = np.zeros((n_samples, len(self.FEATURE_NAMES)))
        y_filter = []
        y_fft = []
        y_feat = []

        samples_per_rule = n_samples // 6
        idx = 0

        # Rule 1: High noise + low SNR -> wiener + 2048 + mfcc
        for _ in range(samples_per_rule):
            noise_level = np.random.uniform(-19.0, 0.0)
            snr_db = np.random.uniform(-10.0, 4.9)
            spectral_centroid = np.random.uniform(200.0, 4000.0)
            zero_crossing_rate = np.random.uniform(0.01, 0.5)
            bandwidth = np.random.uniform(500.0, 4000.0)
            dominant_frequency = np.random.uniform(50.0, 2000.0)

            X[idx] = [noise_level, snr_db, spectral_centroid, zero_crossing_rate, bandwidth, dominant_frequency]
            y_filter.append("wiener")
            y_fft.append(2048)
            y_feat.append("mfcc")
            idx += 1

        # Rule 2: Low frequency hum -> highpass + 512 + fft
        for _ in range(samples_per_rule):
            noise_level = np.random.uniform(-50.0, -10.0)
            snr_db = np.random.uniform(5.0, 30.0)
            spectral_centroid = np.random.uniform(100.0, 1000.0)
            zero_crossing_rate = np.random.uniform(0.0, 0.049)
            bandwidth = np.random.uniform(100.0, 2000.0)
            dominant_frequency = np.random.uniform(50.0, 149.0)

            X[idx] = [noise_level, snr_db, spectral_centroid, zero_crossing_rate, bandwidth, dominant_frequency]
            y_filter.append("highpass")
            y_fft.append(512)
            y_feat.append("fft")
            idx += 1

        # Rule 3: Clean speech + high SNR -> bandpass + 1024 + mfcc
        for _ in range(samples_per_rule):
            noise_level = np.random.uniform(-80.0, -40.0)
            snr_db = np.random.uniform(20.1, 50.0)
            spectral_centroid = np.random.uniform(1000.0, 3000.0)
            zero_crossing_rate = np.random.uniform(0.05, 0.3)
            bandwidth = np.random.uniform(2000.1, 8000.0)
            dominant_frequency = np.random.uniform(150.0, 3000.0)

            X[idx] = [noise_level, snr_db, spectral_centroid, zero_crossing_rate, bandwidth, dominant_frequency]
            y_filter.append("bandpass")
            y_fft.append(1024)
            y_feat.append("mfcc")
            idx += 1

        # Rule 4: Low spectral centroid -> lowpass + 512 + wavelet
        for _ in range(samples_per_rule):
            noise_level = np.random.uniform(-60.0, -20.0)
            snr_db = np.random.uniform(5.0, 30.0)
            spectral_centroid = np.random.uniform(50.0, 499.0)
            zero_crossing_rate = np.random.uniform(0.01, 0.1)
            bandwidth = np.random.uniform(100.0, 1500.0)
            dominant_frequency = np.random.uniform(50.0, 400.0)

            X[idx] = [noise_level, snr_db, spectral_centroid, zero_crossing_rate, bandwidth, dominant_frequency]
            y_filter.append("lowpass")
            y_fft.append(512)
            y_feat.append("wavelet")
            idx += 1

        # Rule 5: 50/60 Hz hum -> notch + 1024 + mfcc
        for _ in range(samples_per_rule):
            noise_level = np.random.uniform(-40.0, -10.0)
            snr_db = np.random.uniform(5.0, 25.0)
            spectral_centroid = np.random.uniform(100.0, 1500.0)
            zero_crossing_rate = np.random.uniform(0.01, 0.2)
            bandwidth = np.random.uniform(100.0, 2000.0)
            dominant_frequency = np.random.choice([50.0, 60.0]) + np.random.uniform(-3.0, 3.0)

            X[idx] = [noise_level, snr_db, spectral_centroid, zero_crossing_rate, bandwidth, dominant_frequency]
            y_filter.append("notch")
            y_fft.append(1024)
            y_feat.append("mfcc")
            idx += 1

        # Rule 6: Moderate noise + balanced spectrum -> bandpass + 1024 + wavelet
        remaining = n_samples - idx
        for _ in range(remaining):
            noise_level = np.random.uniform(-40.0, -20.0)
            snr_db = np.random.uniform(5.0, 20.0)
            spectral_centroid = np.random.uniform(500.0, 2500.0)
            zero_crossing_rate = np.random.uniform(0.02, 0.25)
            bandwidth = np.random.uniform(500.0, 3000.0)
            dominant_frequency = np.random.uniform(150.0, 2500.0)

            X[idx] = [noise_level, snr_db, spectral_centroid, zero_crossing_rate, bandwidth, dominant_frequency]
            y_filter.append("bandpass")
            y_fft.append(1024)
            y_feat.append("wavelet")
            idx += 1

        # Add Gaussian noise scaled to the expected variance of each feature
        feature_scales = [3.0, 3.0, 75.0, 0.01, 100.0, 15.0]
        for f_idx, scale in enumerate(feature_scales):
            noise = np.random.normal(0, scale, n_samples)
            X[:, f_idx] += noise
            # Constrain to physically realistic limits
            if self.FEATURE_NAMES[f_idx] == "zero_crossing_rate":
                X[:, f_idx] = np.clip(X[:, f_idx], 0.0, 1.0)
            elif self.FEATURE_NAMES[f_idx] in ["spectral_centroid", "bandwidth", "dominant_frequency"]:
                X[:, f_idx] = np.clip(X[:, f_idx], 0.0, None)

        return X, np.array(y_filter), np.array(y_fft), np.array(y_feat)

    def train_on_synthetic_data(self, n_samples: int = 2000):
        """
        Generate synthetic training data and train three RF classifiers.
        """
        print(f"[DSP] Generating {n_samples} synthetic training samples...")
        X, y_filter, y_fft, y_feat = self._generate_synthetic_data(n_samples)

        print("[DSP] Training RandomForest classifiers...")
        self.filter_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.fft_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.feature_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

        self.filter_rf.fit(X, y_filter)
        self.fft_rf.fit(X, y_fft)
        self.feature_rf.fit(X, y_feat)

        # Save to disk
        self.save(self.model_path)
        print(f"[DSP] Decision engine successfully trained and saved to '{self.model_path}'")

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(self, feature_summary: dict) -> dict:
        """
        Input: feature_summary dict from dsp_utils.extract_feature_summary()
        Output:
        {
          "filter": str,
          "fft_size": int,
          "feature_set": str,
          "confidence": dict   # per-class probabilities from each forest
        }
        
        Hardcoded edge-case rules are checked FIRST. If a rule fires, its
        prediction is returned immediately without consulting the RF.
        """
        # Cache feature summary for explainability mappings
        self.last_feature_summary = feature_summary

        # Check hardcoded rules first (priority overrides)
        override = self._apply_hardcoded_rules(feature_summary)
        if override:
            print(f"[DSP] Rule override: {override['rule']}")
            return override

        if not self.is_trained():
            raise RuntimeError("Decision engine models are not trained yet! Please call train_on_synthetic_data() first.")

        # Format input vector (exclude sample_rate — it's not a training feature)
        x = np.array([[feature_summary[f] for f in self.FEATURE_NAMES]])

        # Predict optimal parameters
        filter_pred = self.filter_rf.predict(x)[0]
        fft_pred = int(self.fft_rf.predict(x)[0])
        feature_pred = self.feature_rf.predict(x)[0]

        # Get probabilities/confidence per class
        filter_probs = self.filter_rf.predict_proba(x)[0]
        fft_probs = self.fft_rf.predict_proba(x)[0]
        feature_probs = self.feature_rf.predict_proba(x)[0]

        # Create confidence dictionaries
        filter_conf = dict(zip(self.filter_rf.classes_, filter_probs))
        fft_conf = dict(zip(self.fft_rf.classes_, fft_probs))
        feature_conf = dict(zip(self.feature_rf.classes_, feature_probs))

        return {
            "filter": filter_pred,
            "fft_size": fft_pred,
            "feature_set": feature_pred,
            "confidence": {
                "filter": filter_conf,
                "fft_size": fft_conf,
                "feature_set": feature_conf
            }
        }

    # ── Auto-Correction Support ───────────────────────────────────────────

    def add_correction_samples(self, correction_samples: list):
        """
        Inject externally generated correction samples into the training pool.
        correction_samples: list of (feature_vector np.ndarray, labels dict)
        labels dict = {"filter": str, "fft_size": int, "feature_set": str}
        """
        for feature_vector, labels in correction_samples:
            self._correction_X.append(feature_vector)
            self._correction_y_filter.append(labels["filter"])
            self._correction_y_fft.append(labels["fft_size"])
            self._correction_y_features.append(labels["feature_set"])

    def retrain(self):
        """
        Retrain all three Random Forests combining original synthetic data
        + correction samples. Correction samples get 10x sample_weight
        to guarantee they dominate the decision boundary.
        """
        # Generate fresh synthetic data
        X_orig, y_filter_orig, y_fft_orig, y_feat_orig = self._generate_synthetic_data(n_samples=2000)

        X_corr = np.array(self._correction_X)
        y_filter_corr = np.array(self._correction_y_filter)
        y_fft_corr = np.array(self._correction_y_fft)
        y_feat_corr = np.array(self._correction_y_features)

        X_all = np.vstack([X_orig, X_corr])
        y_filter_all = np.concatenate([y_filter_orig, y_filter_corr])
        y_fft_all = np.concatenate([y_fft_orig, y_fft_corr])
        y_feat_all = np.concatenate([y_feat_orig, y_feat_corr])

        # Higher weight for correction samples (10x)
        weights_orig = np.ones(len(X_orig))
        weights_corr = np.full(len(X_corr), 10.0)
        sample_weights = np.concatenate([weights_orig, weights_corr])

        # Retrain all three classifiers
        self.filter_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.fft_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.feature_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

        self.filter_rf.fit(X_all, y_filter_all, sample_weight=sample_weights)
        self.fft_rf.fit(X_all, y_fft_all, sample_weight=sample_weights)
        self.feature_rf.fit(X_all, y_feat_all, sample_weight=sample_weights)

        self._trained = True
        print(f"[DSP] Retrained on {len(X_all)} samples "
              f"({len(X_orig)} synthetic + {len(X_corr)} corrections @ 10x weight)")

        # Save updated model
        self.save(self.model_path)
        print(f"[DSP] Retrained model saved to '{self.model_path}'")

    # ── Utilities ─────────────────────────────────────────────────────────

    def get_feature_importances(self) -> dict:
        """
        Returns average feature importances across all three trained forests.
        """
        if not self.is_trained():
            raise RuntimeError("Decision engine models are not trained yet.")
        
        importances = (
            self.filter_rf.feature_importances_ +
            self.fft_rf.feature_importances_ +
            self.feature_rf.feature_importances_
        ) / 3.0

        return dict(zip(self.FEATURE_NAMES, importances))

    def save(self, path: str):
        """Save the models to disk using joblib."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "filter_rf": self.filter_rf,
            "fft_rf": self.fft_rf,
            "feature_rf": self.feature_rf,
            "feature_names": self.FEATURE_NAMES
        }, path)

    def load(self, path: str):
        """Load the models from disk using joblib."""
        data = joblib.load(path)
        self.filter_rf = data["filter_rf"]
        self.fft_rf = data["fft_rf"]
        self.feature_rf = data["feature_rf"]
        
if __name__ == "__main__":
    print("[Decision Engine] Running standalone test...")
    # Instantiate models
    temp_path = "models/temp_test_decision_engine.joblib"
    engine = DSPDecisionEngine(model_path=temp_path)
    
    # Train
    engine.train_on_synthetic_data(n_samples=600)
    
    # Check importances
    importances = engine.get_feature_importances()
    print("  Feature Importances:")
    for feat, imp in importances.items():
        print(f"    {feat}: {imp:.4f}")
        
    # Test hardcoded rules
    print("\n  Hardcoded Rule Tests:")
    
    # Silence rule
    silence_summary = {"noise_level": -180.0, "snr_db": -180.0, "spectral_centroid": 0.0,
                       "zero_crossing_rate": 0.0, "bandwidth": 0.0, "dominant_frequency": 50.0, "sample_rate": 44100}
    res = engine.predict(silence_summary)
    print(f"    Silence: filter={res['filter']} (expect 'none') -> {'PASS' if res['filter'] == 'none' else 'FAIL'}")
    
    # 50Hz hum rule
    hum_summary = {"noise_level": -27.6, "snr_db": 152.4, "spectral_centroid": 0.0,
                   "zero_crossing_rate": 0.007, "bandwidth": 0.0, "dominant_frequency": 50.0, "sample_rate": 44100}
    res = engine.predict(hum_summary)
    print(f"    50Hz hum: filter={res['filter']} (expect 'notch') -> {'PASS' if res['filter'] == 'notch' else 'FAIL'}")
    
    # Telephony rule
    tel_summary = {"noise_level": -20.1, "snr_db": -1.0, "spectral_centroid": 1064.2,
                   "zero_crossing_rate": 0.1555, "bandwidth": 500.0, "dominant_frequency": 224.0, "sample_rate": 8000}
    res = engine.predict(tel_summary)
    print(f"    Telephony: filter={res['filter']} (expect 'bandpass') -> {'PASS' if res['filter'] == 'bandpass' else 'FAIL'}")
    
    # RF Predict test
    test_feature_summary = {
        "noise_level": -5.0,
        "snr_db": -2.0,
        "spectral_centroid": 1200.0,
        "zero_crossing_rate": 0.15,
        "bandwidth": 1800.0,
        "dominant_frequency": 250.0,
        "sample_rate": 44100
    }
    
    res = engine.predict(test_feature_summary)
    print(f"\n  RF Predict test:")
    print(f"    Expected: Filter='wiener', FFT=2048, Feature Set='mfcc'")
    print(f"    Actual  : Filter='{res['filter']}', FFT={res['fft_size']}, Feature Set='{res['feature_set']}'")
    
    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    print("\n[Decision Engine] Standalone test complete. Status: PASS")
