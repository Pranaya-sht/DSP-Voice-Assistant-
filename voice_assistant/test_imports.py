"""Quick integration test — verifies all project modules load correctly."""
import sys
import os

# Fix Windows console encoding for emoji support
if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

from audio_utils import record_audio, save_wav, print_audio_stats
from dsp_utils import (compute_fft, plot_waveform, plot_fft,
                        get_dominant_frequency, compute_rms_energy)
from ai_utils import (query_ai, ConversationHistory, check_wake_word,
                       handle_special_commands)
from tts_utils import speak_response, clean_text_for_speech

print("=== ALL PROJECT MODULES LOADED OK ===")

# Test ConversationHistory
h = ConversationHistory()
print(f"ConversationHistory created, turns: {h.turn_count}")

# Test wake word detection
result = check_wake_word("hello assistant how are you")
print(f"Wake word test ('hello assistant'): {result}")

# Test text cleaning
cleaned = clean_text_for_speech("**Bold** and *italic* with `code`")
print(f'Clean text test: "{cleaned}"')

# Test special commands
cmd = handle_special_commands("quit", h)
print(f"Special command 'quit': {cmd}")

# Test DSP functions with synthetic data
import numpy as np
sample_rate = 44100
t = np.linspace(0, 1, sample_rate, endpoint=False)
# Generate a 440 Hz sine wave (concert A note)
test_signal = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

freqs, mags = compute_fft(test_signal, sample_rate)
dom_freq, dom_mag = get_dominant_frequency(freqs, mags)
rms = compute_rms_energy(test_signal)
print(f"\nDSP test (440 Hz sine wave):")
print(f"  Dominant frequency: {dom_freq:.1f} Hz (expected ~440)")
print(f"  RMS energy: {rms:.4f}")
print(f"  Samples: {len(test_signal):,}")

print("\n=== ALL TESTS PASSED ===")
