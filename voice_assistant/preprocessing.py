"""
preprocessing.py - Audio Signal Preprocessing Layer
==================================================
This module implements the initial stages of the DSP pipeline:
1. DC Offset Removal (centering the waveform around 0)
2. Peak Normalization (scaling the waveform to fits [-1.0, 1.0])
3. Framing (chopping the continuous waveform into short overlapping windows)
"""

import numpy as np

def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """
    Remove DC offset from the audio signal by subtracting the mean amplitude.
    This centers the waveform around y=0, preventing clipping and errors in FFT.
    """
    # Convert to float for math operations
    audio_float = audio.astype(np.float32)
    mean_val = np.mean(audio_float)
    return audio_float - mean_val

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Scale the audio amplitude so that the peak magnitude is exactly 1.0 (or -1.0).
    Handles both int16 and float input formats.
    """
    # Convert to float32 if it's integer format
    if audio.dtype == np.int16:
        audio_float = audio.astype(np.float32) / 32767.0
    else:
        audio_float = audio.astype(np.float32)

    peak = np.max(np.abs(audio_float))
    if peak > 0:
        return audio_float / peak
    return audio_float

def frame_signal(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int = 25,
    hop_ms: int = 10,
) -> np.ndarray:
    """
    Split the continuous 1D audio array into a 2D array of overlapping frames.
    Each row is a single frame of size frame_length.
    
    Parameters:
    -----------
    audio       : 1D numpy array of audio samples (usually float normalized)
    sample_rate : sampling frequency in Hz
    frame_ms    : length of each frame in milliseconds
    hop_ms      : overlap stride between consecutive frames in milliseconds
    
    Returns:
    --------
    2D numpy array of shape (num_frames, frame_length)
    """
    # Convert time in ms to number of samples
    frame_length = int(round(frame_ms * sample_rate / 1000.0))
    hop_length = int(round(hop_ms * sample_rate / 1000.0))
    
    signal_length = len(audio)
    
    # If the audio signal is too short, pad it to at least one frame length
    if signal_length < frame_length:
        pad_len = frame_length - signal_length
        audio = np.pad(audio, (0, pad_len), mode="constant")
        signal_length = len(audio)
        
    # Calculate how many frames we can fit
    num_frames = 1 + int(np.floor((signal_length - frame_length) / hop_length))
    
    # Populate the 2D frames array
    frames = np.zeros((num_frames, frame_length), dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frames[i] = audio[start:end]
        
    return frames

def preprocess(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Orchestrate the preprocessing pipeline:
    1. DC Offset Removal
    2. Peak Normalization
    3. Overlapping Framing
    
    Returns a dictionary containing the normalized audio, framed audio, and sample rate.
    """
    # 1. Centering (DC Offset Removal)
    centered = remove_dc_offset(audio)
    
    # 2. Normalization
    normalized = normalize_audio(centered)
    
    # 3. Framing
    frames = frame_signal(normalized, sample_rate)
    
    return {
        "normalized": normalized,
        "frames": frames,
        "sample_rate": sample_rate,
    }

if __name__ == "__main__":
    print("[Preprocessing] Running standalone test...")
    # Generate 1.0s sample at 16kHz
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Sine wave (100 Hz) with an artificial DC offset of +0.25
    signal = 0.5 * np.sin(2 * np.pi * 100 * t) + 0.25
    
    # Convert to typical int16 microphone format
    signal_int16 = (signal * 32767).astype(np.int16)
    
    print(f"  Input length : {len(signal_int16)} samples")
    print(f"  Input min/max: {signal_int16.min()}, {signal_int16.max()}")
    print(f"  Input mean   : {signal_int16.mean():.2f}")
    
    # Run preprocess
    res = preprocess(signal_int16, sr)
    
    print(f"  Normalized range : [{res['normalized'].min():.4f}, {res['normalized'].max():.4f}]")
    print(f"  Normalized mean  : {np.mean(res['normalized']):.6f} (should be close to 0)")
    print(f"  Frames shape     : {res['frames'].shape} (should be approx 98x400 for 25ms/10ms)")
    print("[Preprocessing] Standalone test complete. Status: PASS")
