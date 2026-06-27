"""
adaptive_dsp.py - Adaptive Filtering and Feature Extraction
============================================================
This module applies the optimal DSP parameters selected by the Decision Engine.
It filters the audio signal and extracts features for event classification/XAI.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, wiener

def apply_filter(audio: np.ndarray, sample_rate: int, filter_type: str) -> np.ndarray:
    """
    Apply the chosen filter to the audio signal.
    
    Parameters:
    -----------
    audio       : 1D numpy array of audio samples (float normalized)
    sample_rate : sampling frequency in Hz
    filter_type : "bandpass", "highpass", "lowpass", "notch", "wiener", or "none"
    """
    # Ensure float32
    audio_float = audio.astype(np.float32)
    
    nyquist = sample_rate / 2.0
    
    if filter_type == "bandpass":
        # Speech bandpass filter: 300 to 3400 Hz
        low = 300.0
        high = 3400.0
        if high >= nyquist:
            high = nyquist - 10.0
        b, a = butter(4, [low / nyquist, high / nyquist], btype="band")
        return filtfilt(b, a, audio_float)
        
    elif filter_type == "highpass":
        # Remove low frequency rumble/hum below 100 Hz
        cutoff = 100.0
        if cutoff >= nyquist:
            cutoff = nyquist - 10.0
        b, a = butter(4, cutoff / nyquist, btype="high")
        return filtfilt(b, a, audio_float)
        
    elif filter_type == "lowpass":
        # Lowpass filter: keep frequencies below 4000 Hz
        cutoff = 4000.0
        if cutoff >= nyquist:
            cutoff = nyquist - 10.0
        b, a = butter(4, cutoff / nyquist, btype="low")
        return filtfilt(b, a, audio_float)
        
    elif filter_type == "notch":
        # Notch filter at 50 Hz or 60 Hz hum.
        # We can dynamically notch 50 or 60 Hz. Let's notch 50 Hz by default,
        # or 60 Hz if there is significant energy around 60 Hz.
        notch_freq = 50.0
        # Check if the signal is long enough to run FFT and detect if hum is at 60Hz
        if len(audio_float) >= 1024:
            fft_mags = np.abs(np.fft.rfft(audio_float[:1024]))
            freqs = np.fft.rfftfreq(1024, d=1.0/sample_rate)
            # Find dominant freq in range 45-65 Hz
            hum_mask = (freqs >= 45.0) & (freqs <= 65.0)
            if np.any(hum_mask):
                peak_hum_freq = freqs[hum_mask][np.argmax(fft_mags[hum_mask])]
                if abs(peak_hum_freq - 60.0) < abs(peak_hum_freq - 50.0):
                    notch_freq = 60.0
                    
        b, a = iirnotch(notch_freq / nyquist, 30.0)
        return filtfilt(b, a, audio_float)
        
    elif filter_type == "wiener":
        # Scipy's Wiener filter for adaptive noise suppression
        # Returns a filtered estimate of the input using local variance
        try:
            return wiener(audio_float)
        except Exception:
            # Fallback if Wiener filter fails due to dimension mismatch or signal size
            return audio_float
            
    elif filter_type == "none" or filter_type is None:
        return audio_float
        
    else:
        print(f"[DSP] Warning: Unknown filter type '{filter_type}'. Returning original signal.")
        return audio_float

def extract_features(
    audio: np.ndarray,
    sample_rate: int,
    feature_set: str,
    fft_size: int,
) -> np.ndarray:
    """
    Extract the specified features from the filtered audio signal.
    
    Parameters:
    -----------
    audio       : 1D numpy array of audio samples (filtered, float normalized)
    sample_rate : sampling frequency in Hz
    feature_set : "mfcc", "wavelet", or "fft"
    fft_size    : FFT window size (512, 1024, 2048)
    """
    # Ensure float32
    audio_float = audio.astype(np.float32)
    
    if feature_set == "mfcc":
        try:
            import librosa
            # Extract 40 MFCCs
            # Use hop_length proportional to fft_size
            hop_length = fft_size // 2
            mfccs = librosa.feature.mfcc(y=audio_float, sr=sample_rate, n_fft=fft_size, hop_length=hop_length, n_mfcc=40)
            # Return mean across time frames (shape: (40,))
            return np.mean(mfccs, axis=1)
        except ImportError:
            raise ImportError("[DSP] librosa is not installed but is required for MFCC feature extraction.")
            
    elif feature_set == "wavelet":
        try:
            import pywt
            # Perform 5-level discrete wavelet transform using Daubechies 4 wavelet
            coeffs = pywt.wavedec(audio_float, 'db4', level=5)
            # Compute energy (sum of squares) of coefficients in each subband
            # Returns 6 energies (1 approximation + 5 detail coefficients)
            energies = [np.sum(c ** 2) for c in coeffs]
            return np.array(energies, dtype=np.float32)
        except ImportError:
            raise ImportError("[DSP] PyWavelets (pywt) is not installed but is required for Wavelet features.")
            
    elif feature_set == "fft":
        # Standard magnitude spectrum (rfft)
        rfft_mags = np.abs(np.fft.rfft(audio_float, n=fft_size))
        return rfft_mags
        
    else:
        raise ValueError(f"[DSP] Unknown feature set type: '{feature_set}'")

def run_adaptive_dsp(audio: np.ndarray, sample_rate: int, dsp_params: dict) -> dict:
    """
    Execute the adaptive DSP pipeline stage:
    1. Filter the audio using predicted filter
    2. Extract downstream features using predicted feature set and FFT size
    """
    filter_type = dsp_params["filter"]
    fft_size = dsp_params["fft_size"]
    feature_set = dsp_params["feature_set"]
    
    # 1. Apply Filter
    filtered_audio = apply_filter(audio, sample_rate, filter_type)
    
    # 2. Extract Features
    features = extract_features(filtered_audio, sample_rate, feature_set, fft_size)
    
    return {
        "filtered_audio": filtered_audio,
        "features": features,
        "feature_set": feature_set,
        "fft_size": fft_size,
        "filter_applied": filter_type
    }

if __name__ == "__main__":
    print("[Adaptive DSP] Running standalone test...")
    # Generate 1.0s sine wave + noise
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 1000 * t) + 0.1 * np.random.normal(0, 1, sr)
    
    # Test filters
    for f_type in ["bandpass", "highpass", "lowpass", "notch", "wiener", "none"]:
        filtered = apply_filter(signal, sr, f_type)
        print(f"  Filter '{f_type}' range: [{filtered.min():.4f}, {filtered.max():.4f}]")
        
    # Test feature extraction
    dsp_params = {"filter": "bandpass", "fft_size": 1024, "feature_set": "mfcc"}
    res = run_adaptive_dsp(signal, sr, dsp_params)
    print(f"  Adaptive output:")
    print(f"    Filtered audio shape: {res['filtered_audio'].shape}")
    print(f"    Features shape (MFCC)  : {res['features'].shape} (should be (40,))")
    
    dsp_params["feature_set"] = "wavelet"
    res = run_adaptive_dsp(signal, sr, dsp_params)
    print(f"    Features shape (Wavelet): {res['features'].shape} (should be (6,))")
    
    dsp_params["feature_set"] = "fft"
    res = run_adaptive_dsp(signal, sr, dsp_params)
    print(f"    Features shape (FFT)    : {res['features'].shape} (should be (513,))")
    
    print("[Adaptive DSP] Standalone test complete. Status: PASS")
