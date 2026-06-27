import numpy as np
from scipy.io import wavfile

# sr = 44100
# t = np.linspace(0, 5, sr * 5)
# hum = 0.3 * np.sin(2 * np.pi * 50 * t)       # 50 Hz hum
# speech = 0.7 * np.random.randn(sr * 5) * 0.1  # placeholder speech noise
# wavfile.write("test_hum.wav", sr, (hum + speech).astype(np.float32))


wavfile.write("test_silence.wav", 44100, np.zeros(44100 * 3, dtype=np.float32))