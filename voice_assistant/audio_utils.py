"""
audio_utils.py - Audio Recording and WAV File Utilities
========================================================
This module handles all microphone recording and audio file operations.

Key concepts for beginners:
- sounddevice: Python library to record/play audio using your sound card
- numpy: Library for numerical arrays (audio is just an array of numbers!)
- scipy.io.wavfile: Library to read/write WAV audio files
- Sample rate: How many audio samples captured per second (44100 Hz is CD quality)
- Dtype: The data type of each sample (int16 = 16-bit integer, common for audio)
"""

import sounddevice as sd      # For microphone access
import numpy as np             # For numerical operations on audio data
import scipy.io.wavfile as wav # For saving/loading WAV files
import os                      # For file path operations
import time                    # For timing recordings
from typing import Optional, Tuple  # For type hints (improves code readability)


# ─── Constants ────────────────────────────────────────────────────────────────
# These are fixed values used throughout the module.
# Using UPPER_CASE is a Python convention for constants.

DEFAULT_SAMPLE_RATE: int = 44100   # 44100 samples per second (CD quality)
DEFAULT_CHANNELS: int = 1          # Mono audio (1 channel). Stereo would be 2.
DEFAULT_DTYPE: str = "int16"       # 16-bit integer samples (-32768 to 32767)
DEFAULT_DURATION: int = 5          # Default recording duration in seconds
OUTPUT_DIR: str = "recordings"     # Folder where WAV files are saved


# ─── Setup ────────────────────────────────────────────────────────────────────

def ensure_output_dir() -> None:
    """
    Create the recordings directory if it doesn't already exist.
    
    os.makedirs with exist_ok=True won't raise an error if the folder exists.
    This is safer than checking first (avoids race conditions).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[Audio] Output directory ready: '{OUTPUT_DIR}/'")


# ─── Microphone Listing ───────────────────────────────────────────────────────

def list_microphones() -> None:
    """
    Print all available audio input devices to help the user choose.
    
    sounddevice.query_devices() returns info about all audio devices.
    We filter for 'max_input_channels > 0' to find microphones.
    """
    print("\n[Audio] Available microphone devices:")
    print("-" * 50)
    
    # sd.query_devices() returns a list of device dictionaries
    devices = sd.query_devices()
    
    for i, device in enumerate(devices):
        # Only show input devices (microphones), not output devices (speakers)
        if device["max_input_channels"] > 0:
            print(f"  [{i}] {device['name']} "
                  f"(channels: {device['max_input_channels']}, "
                  f"rate: {device['default_samplerate']:.0f} Hz)")
    
    print("-" * 50)


# ─── Core Recording Function ──────────────────────────────────────────────────

def record_audio(
    duration: int = DEFAULT_DURATION,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    dtype: str = DEFAULT_DTYPE,
    device: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Record audio from the microphone for a given duration.

    Parameters:
    -----------
    duration    : int   - How many seconds to record
    sample_rate : int   - Samples per second (Hz). Higher = better quality.
    channels    : int   - 1 for mono, 2 for stereo
    dtype       : str   - Data type for samples ('int16', 'float32', etc.)
    device      : int   - Specific device index (None = system default)

    Returns:
    --------
    Tuple of (audio_data, sample_rate)
    - audio_data : numpy array of shape (num_samples, channels) or (num_samples,)
    - sample_rate : the rate used (needed for WAV saving and DSP analysis)

    How sd.rec() works:
    - frames = total samples to capture = duration × sample_rate
    - It returns immediately and records in the background
    - sd.wait() blocks until recording is complete
    """
    ensure_output_dir()
    
    total_frames = int(duration * sample_rate)  # e.g., 5s × 44100 = 220500 samples
    
    print(f"\n[Audio] 🎙️  Recording for {duration} second(s)...")
    print(f"[Audio]    Sample rate : {sample_rate} Hz")
    print(f"[Audio]    Channels    : {channels}")
    print(f"[Audio]    Data type   : {dtype}")
    
    try:
        # sd.rec() starts recording in the background
        # It returns a numpy array that gets filled as audio comes in
        audio_data: np.ndarray = sd.rec(
            frames=total_frames,
            samplerate=sample_rate,
            channels=channels,
            dtype=dtype,
            device=device,  # None means use system default microphone
        )
        
        # sd.wait() blocks (pauses the program) until all frames are captured
        sd.wait()
        
        print(f"[Audio] ✅ Recording complete! Captured {len(audio_data)} samples.")
        
        # If mono (1 channel), flatten from shape (N,1) to (N,) for easier processing
        if channels == 1:
            audio_data = audio_data.flatten()
        
        return audio_data, sample_rate

    except sd.PortAudioError as e:
        # PortAudio is the underlying C library; this catches device errors
        print(f"[Audio] ❌ PortAudio error: {e}")
        print("[Audio]    Check if your microphone is connected and not in use.")
        raise
    except Exception as e:
        print(f"[Audio] ❌ Unexpected error during recording: {e}")
        raise


# ─── WAV File Saving ──────────────────────────────────────────────────────────

def save_wav(
    audio_data: np.ndarray,
    sample_rate: int,
    filename: str = "recording.wav",
) -> str:
    """
    Save a numpy audio array to a WAV file on disk.

    Parameters:
    -----------
    audio_data  : numpy array of audio samples
    sample_rate : int - the sample rate used during recording
    filename    : str - name for the output file (saved in OUTPUT_DIR)

    Returns:
    --------
    str - the full file path of the saved WAV file

    WAV format:
    - Standard uncompressed audio format
    - Supported by virtually all audio software
    - scipy.io.wavfile handles the binary header and data automatically
    """
    ensure_output_dir()
    
    # Build full path: e.g., "recordings/recording.wav"
    filepath: str = os.path.join(OUTPUT_DIR, filename)
    
    try:
        # scipy.io.wavfile.write(filename, rate, data)
        # - rate: sample rate in Hz
        # - data: numpy array (int16 or float32)
        wav.write(filepath, sample_rate, audio_data)
        
        # Calculate file size for feedback
        file_size_kb = os.path.getsize(filepath) / 1024
        duration_s = len(audio_data) / sample_rate
        
        print(f"[Audio] 💾 WAV saved: '{filepath}'")
        print(f"[Audio]    Duration : {duration_s:.2f} seconds")
        print(f"[Audio]    File size: {file_size_kb:.1f} KB")
        
        return filepath

    except PermissionError:
        print(f"[Audio] ❌ Permission denied writing to '{filepath}'.")
        print("[Audio]    Make sure the file is not open in another program.")
        raise
    except Exception as e:
        print(f"[Audio] ❌ Failed to save WAV: {e}")
        raise


# ─── WAV File Loading ─────────────────────────────────────────────────────────

def load_wav(filepath: str) -> Tuple[np.ndarray, int]:
    """
    Load a WAV file from disk into a numpy array.

    Parameters:
    -----------
    filepath : str - full path to the WAV file

    Returns:
    --------
    Tuple of (audio_data, sample_rate)

    Useful for:
    - Reloading a saved recording for analysis
    - Testing DSP functions with pre-recorded audio
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[Audio] WAV file not found: '{filepath}'")
    
    try:
        # scipy.io.wavfile.read() returns (sample_rate, data_array)
        sample_rate, audio_data = wav.read(filepath)
        
        print(f"[Audio] 📂 Loaded WAV: '{filepath}'")
        print(f"[Audio]    Sample rate: {sample_rate} Hz")
        print(f"[Audio]    Samples    : {len(audio_data)}")
        print(f"[Audio]    Duration   : {len(audio_data)/sample_rate:.2f}s")
        
        return audio_data, sample_rate

    except Exception as e:
        print(f"[Audio] ❌ Failed to load WAV '{filepath}': {e}")
        raise


# ─── Audio Statistics ─────────────────────────────────────────────────────────

def get_audio_stats(audio_data: np.ndarray, sample_rate: int) -> dict:
    """
    Compute basic statistics about an audio signal.
    
    Returns a dictionary with:
    - duration_s   : length in seconds
    - num_samples  : total sample count
    - sample_rate  : Hz
    - max_amplitude: loudest sample value
    - rms_energy   : Root Mean Square energy (perceived loudness)
    - db_level     : decibel level (20*log10(rms))
    
    RMS (Root Mean Square) = sqrt(mean(sample²))
    It's a measure of signal power, closely related to perceived loudness.
    dB = 20 * log10(rms / max_possible) — standard audio loudness measure.
    """
    # Normalize to float in range [-1, 1] for consistent math
    # int16 max value is 32767
    if audio_data.dtype == np.int16:
        float_audio = audio_data.astype(np.float32) / 32767.0
    else:
        float_audio = audio_data.astype(np.float32)
    
    rms = float(np.sqrt(np.mean(float_audio ** 2)))  # RMS energy
    max_amp = float(np.max(np.abs(float_audio)))       # Peak amplitude
    
    # Avoid log(0) — add tiny epsilon
    epsilon = 1e-10
    db_level = 20 * np.log10(rms + epsilon)
    
    return {
        "duration_s"   : len(audio_data) / sample_rate,
        "num_samples"  : len(audio_data),
        "sample_rate"  : sample_rate,
        "max_amplitude": max_amp,
        "rms_energy"   : rms,
        "db_level"     : float(db_level),
    }


def print_audio_stats(audio_data: np.ndarray, sample_rate: int) -> None:
    """Print audio statistics in a formatted table."""
    stats = get_audio_stats(audio_data, sample_rate)
    print("\n[Audio] 📊 Audio Statistics:")
    print(f"         Duration    : {stats['duration_s']:.2f} s")
    print(f"         Samples     : {stats['num_samples']:,}")
    print(f"         Sample Rate : {stats['sample_rate']} Hz")
    print(f"         Max Amp     : {stats['max_amplitude']:.4f}")
    print(f"         RMS Energy  : {stats['rms_energy']:.4f}")
    print(f"         Level (dB)  : {stats['db_level']:.1f} dB")


# ─── Noise Estimation ─────────────────────────────────────────────────────────

def estimate_noise_level(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    noise_duration: float = 1.0,
) -> float:
    """
    Record a short silence to estimate background noise level (RMS).
    
    This is useful for voice activity detection (VAD):
    - Record 1 second of silence
    - Compute RMS of that silence
    - Any audio louder than (noise_rms * threshold) is speech
    
    Parameters:
    -----------
    sample_rate    : int   - sample rate to use
    noise_duration : float - seconds to record for noise estimation (default 1s)
    
    Returns:
    --------
    float - RMS noise level (between 0.0 and 1.0)
    """
    print(f"\n[Audio] 🔇 Estimating background noise ({noise_duration}s)...")
    print("[Audio]    Please stay quiet...")
    
    noise_data, _ = record_audio(
        duration=int(noise_duration),
        sample_rate=sample_rate,
    )
    
    # Normalize and compute RMS
    if noise_data.dtype == np.int16:
        noise_float = noise_data.astype(np.float32) / 32767.0
    else:
        noise_float = noise_data.astype(np.float32)
    
    noise_rms = float(np.sqrt(np.mean(noise_float ** 2)))
    print(f"[Audio]    Noise RMS: {noise_rms:.6f} ({20*np.log10(noise_rms+1e-10):.1f} dB)")
    
    return noise_rms
