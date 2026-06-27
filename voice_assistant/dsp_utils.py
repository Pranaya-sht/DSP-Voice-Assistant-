"""
dsp_utils.py - Digital Signal Processing & Visualization
=========================================================
This module implements all DSP analysis and real-time plotting functions.

Key DSP concepts for beginners:
- FFT (Fast Fourier Transform): Converts a time-domain signal (waveform)
  into the frequency domain — showing WHICH frequencies are present and 
  HOW LOUD each one is.
  
- Time domain: Audio samples plotted over time → the waveform you see in 
  audio editors.
  
- Frequency domain: Same audio plotted by frequency → shows if it's bass 
  (low Hz), mid, or treble (high Hz). Voice is typically 85-255 Hz for 
  fundamental, 300-3400 Hz for speech intelligibility.

- Dominant frequency: The frequency with the highest energy in the FFT.

- Energy (RMS): How loud the signal is overall.

- Spectrogram: A 2D plot showing how frequency content changes over time.
  Like a "heat map" of audio.
"""

import numpy as np              # For FFT and array math
import matplotlib               # Base plotting library
matplotlib.use("TkAgg")         # Use TkAgg backend (works on most systems)
                                # Change to "Qt5Agg" if you have Qt installed
import matplotlib.pyplot as plt  # High-level plotting interface
import matplotlib.gridspec as gridspec  # For creating complex subplot layouts
import matplotlib.animation as animation  # For real-time animated plots
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, Tuple, List
import threading                # For running real-time plot in background thread


# ─── FFT Analysis ─────────────────────────────────────────────────────────────

def compute_fft(
    audio_data: np.ndarray,
    sample_rate: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Fast Fourier Transform of an audio signal.

    The FFT converts a time-domain signal (amplitude vs time) into
    the frequency domain (amplitude vs frequency).

    Math behind FFT:
    ----------------
    For N samples at sample_rate Hz:
    - The FFT produces N complex numbers
    - We only need the first N//2 (the rest are mirror images — Nyquist theorem)
    - |FFT[k]| = magnitude of frequency k
    - Frequency k corresponds to: f = k * sample_rate / N

    Parameters:
    -----------
    audio_data  : numpy array of audio samples
    sample_rate : int - the sample rate in Hz

    Returns:
    --------
    Tuple of (frequencies, magnitudes)
    - frequencies : 1D array of frequency values in Hz (x-axis)
    - magnitudes  : 1D array of corresponding magnitudes (y-axis)
    """
    n_samples = len(audio_data)  # Total number of samples, e.g., 220500
    
    # Normalize to float [-1, 1] for consistent FFT scaling
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32767.0
    else:
        audio_float = audio_data.astype(np.float32)
    
    # Apply Hanning window to reduce "spectral leakage"
    # Without windowing, sharp edges at start/end of recording create 
    # false high-frequency components in the FFT.
    # The Hanning window tapers the signal to zero at both ends.
    window = np.hanning(n_samples)       # Shape: (n_samples,)
    windowed_audio = audio_float * window  # Element-wise multiplication
    
    # np.fft.fft() computes the FFT
    # Result is complex: each element = a + bi where magnitude = sqrt(a²+b²)
    fft_result = np.fft.fft(windowed_audio)  # Shape: (n_samples,) complex
    
    # np.fft.fftfreq() gives the corresponding frequency for each FFT bin
    # 'd' parameter = time step between samples = 1/sample_rate
    frequencies = np.fft.fftfreq(n_samples, d=1.0/sample_rate)
    
    # Only keep the positive frequencies (first half)
    # Negative frequencies are mirror images and not useful for audio analysis
    n_positive = n_samples // 2
    frequencies = frequencies[:n_positive]       # Frequencies: 0 to sample_rate/2
    magnitudes = np.abs(fft_result[:n_positive]) # Magnitude = |complex number|
    
    # Normalize magnitude by number of samples
    magnitudes = magnitudes / n_samples
    
    return frequencies, magnitudes


def get_dominant_frequency(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    min_freq: float = 50.0,   # Ignore very low frequencies (rumble/noise)
    max_freq: float = 4000.0, # Focus on speech range (up to 4 kHz)
) -> Tuple[float, float]:
    """
    Find the dominant (loudest) frequency in the FFT result.

    Parameters:
    -----------
    frequencies : array of frequency values in Hz
    magnitudes  : array of FFT magnitude values
    min_freq    : minimum frequency to consider (Hz)
    max_freq    : maximum frequency to consider (Hz)

    Returns:
    --------
    Tuple of (dominant_freq_hz, magnitude_at_that_freq)
    
    This is useful for:
    - Identifying the fundamental pitch of a voice
    - Detecting musical notes
    - Checking if background noise is a specific frequency (e.g., 60 Hz hum)
    """
    # Create a boolean mask for the frequency range we care about
    # mask is True where: min_freq <= frequency <= max_freq
    freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
    
    if not np.any(freq_mask):
        # No frequencies in range — return 0
        return 0.0, 0.0
    
    # Apply mask to get only the frequencies/magnitudes in our range
    filtered_freqs = frequencies[freq_mask]
    filtered_mags  = magnitudes[freq_mask]
    
    # np.argmax() returns the INDEX of the maximum value
    peak_idx = np.argmax(filtered_mags)
    
    dominant_freq = float(filtered_freqs[peak_idx])
    dominant_mag  = float(filtered_mags[peak_idx])
    
    return dominant_freq, dominant_mag


# ─── Energy / Loudness ────────────────────────────────────────────────────────

def compute_rms_energy(audio_data: np.ndarray) -> float:
    """
    Compute the RMS (Root Mean Square) energy of an audio signal.

    RMS formula: sqrt( (1/N) * Σ(x_i²) )
    
    - Square each sample: eliminates negatives, emphasizes loud peaks
    - Average: normalize by sample count
    - Square root: undo the squaring to get back to amplitude units
    
    RMS closely matches human perception of loudness (unlike peak amplitude).
    Returns a value between 0.0 (silence) and 1.0 (maximum loudness).
    """
    # Normalize to float [-1, 1]
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32767.0
    else:
        audio_float = audio_data.astype(np.float32)
    
    # np.mean(audio_float ** 2) = mean of squares
    # np.sqrt(...) = root of mean of squares
    rms = float(np.sqrt(np.mean(audio_float ** 2)))
    return rms


def rms_to_db(rms: float) -> float:
    """
    Convert RMS energy to decibels (dB).
    
    Formula: dB = 20 * log10(rms)
    - 0 dB = maximum loudness (rms = 1.0)
    - -60 dB = near silence
    - Every 20 dB = 10× change in RMS
    
    We add a tiny epsilon to avoid log10(0) = -infinity.
    """
    epsilon = 1e-10  # Tiny value to prevent math errors
    return 20.0 * float(np.log10(rms + epsilon))


# ─── Waveform Plot ────────────────────────────────────────────────────────────

def plot_waveform(
    audio_data: np.ndarray,
    sample_rate: int,
    title: str = "Audio Waveform",
    ax: Optional[Axes] = None,
    show: bool = False,
) -> Axes:
    """
    Plot the audio waveform (amplitude vs time).

    The waveform shows how the air pressure (captured by the mic) changes 
    over time. Speech has a characteristic shape: quiet before/after, 
    and active in between.

    Parameters:
    -----------
    audio_data  : numpy array of audio samples
    sample_rate : int - sample rate in Hz (used to calculate time axis)
    title       : str - plot title
    ax          : matplotlib Axes (if None, creates a new figure)
    show        : bool - if True, displays the plot immediately

    Returns:
    --------
    matplotlib Axes object (so you can add more elements to the plot)
    """
    # Create time axis: [0, 1/sr, 2/sr, ..., (N-1)/sr] in seconds
    # np.arange(N) = [0, 1, 2, ..., N-1]
    n_samples = len(audio_data)
    time_axis = np.arange(n_samples) / sample_rate  # Shape: (N,) in seconds
    
    # Normalize audio to float for consistent y-axis scaling
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32767.0
    else:
        audio_float = audio_data.astype(np.float32)
    
    # Create axes if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))
    
    # Plot waveform: thin line for detail, blue-ish color
    ax.plot(time_axis, audio_float, color="#4A90E2", linewidth=0.6, alpha=0.8)
    
    # Draw a horizontal line at y=0 (silence reference)
    ax.axhline(y=0, color="#888888", linewidth=0.5, linestyle="--", alpha=0.5)
    
    # Compute and show RMS as a horizontal line
    rms = compute_rms_energy(audio_data)
    ax.axhline(y=rms, color="#FF6B6B", linewidth=1.2, linestyle="--",
               alpha=0.7, label=f"RMS = {rms:.4f}")
    ax.axhline(y=-rms, color="#FF6B6B", linewidth=1.2, linestyle="--", alpha=0.7)
    
    # Formatting
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Amplitude", fontsize=11)
    ax.set_xlim(0, time_axis[-1])
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_facecolor("#F8F9FA")
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


# ─── FFT Plot ─────────────────────────────────────────────────────────────────

def plot_fft(
    audio_data: np.ndarray,
    sample_rate: int,
    title: str = "FFT Magnitude Spectrum",
    ax: Optional[Axes] = None,
    show: bool = False,
    max_freq_display: float = 8000.0,  # Show up to 8 kHz (covers most speech)
) -> Axes:
    """
    Plot the FFT magnitude spectrum (frequency content of the audio).

    The FFT spectrum shows WHICH frequencies make up the sound:
    - Low frequencies (50-300 Hz): bass, fundamental voice pitch
    - Mid frequencies (300-3400 Hz): speech intelligibility zone
    - High frequencies (3400-8000 Hz): consonants, sibilance (s, sh sounds)

    Parameters:
    -----------
    audio_data       : numpy array of audio samples
    sample_rate      : int - sample rate in Hz
    title            : str - plot title
    ax               : matplotlib Axes (if None, creates new figure)
    show             : bool - show plot immediately
    max_freq_display : float - maximum frequency shown on x-axis (Hz)

    Returns:
    --------
    matplotlib Axes object
    """
    # Compute FFT
    frequencies, magnitudes = compute_fft(audio_data, sample_rate)
    
    # Find dominant frequency in speech range
    dom_freq, dom_mag = get_dominant_frequency(frequencies, magnitudes)
    
    # Filter to display range (e.g., 0 to 8000 Hz)
    display_mask = frequencies <= max_freq_display
    display_freqs = frequencies[display_mask]
    display_mags  = magnitudes[display_mask]
    
    # Create axes if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    
    # Fill under the curve for visual appeal
    ax.fill_between(display_freqs, display_mags, alpha=0.3, color="#7B68EE")
    ax.plot(display_freqs, display_mags, color="#7B68EE", linewidth=1.0)
    
    # Mark the dominant frequency with a vertical line + annotation
    if dom_freq > 0:
        ax.axvline(x=dom_freq, color="#FF6B6B", linewidth=2.0, linestyle="--",
                   label=f"Dominant: {dom_freq:.1f} Hz")
        ax.annotate(
            f"{dom_freq:.0f} Hz",
            xy=(dom_freq, dom_mag),
            xytext=(dom_freq + 100, dom_mag * 0.8),
            fontsize=9, color="#FF6B6B",
            arrowprops=dict(arrowstyle="->", color="#FF6B6B", lw=1.5),
        )
    
    # Shade the speech intelligibility band (300-3400 Hz)
    ax.axvspan(300, 3400, alpha=0.08, color="green", label="Speech band (300-3400 Hz)")
    
    # Formatting
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Magnitude", fontsize=11)
    ax.set_xlim(0, max_freq_display)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_facecolor("#F8F9FA")
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


# ─── Combined Dashboard Plot ──────────────────────────────────────────────────

def plot_audio_dashboard(
    audio_data: np.ndarray,
    sample_rate: int,
    title: str = "Voice Assistant — Audio Analysis",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Create a comprehensive 4-panel dashboard showing:
    1. Waveform (amplitude vs time)
    2. FFT spectrum (magnitude vs frequency)
    3. Energy bar (current RMS energy level with color coding)
    4. Statistics panel (text summary of key metrics)

    This is the main visualization function called after each recording.

    Parameters:
    -----------
    audio_data  : numpy array of audio samples
    sample_rate : int - sample rate in Hz
    title       : str - figure title
    save_path   : str or None - if provided, saves the figure to this path

    Returns:
    --------
    matplotlib Figure object
    """
    # ── Compute all metrics first ──────────────────────────────────────────
    frequencies, magnitudes = compute_fft(audio_data, sample_rate)
    dom_freq, dom_mag = get_dominant_frequency(frequencies, magnitudes)
    rms = compute_rms_energy(audio_data)
    db = rms_to_db(rms)
    duration = len(audio_data) / sample_rate
    
    # ── Create Figure with GridSpec layout ────────────────────────────────
    # GridSpec lets us control relative sizes of each subplot
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    
    # Create grid: 3 rows × 2 columns
    # Row 0: Waveform (spans both columns)
    # Row 1: FFT (spans both columns)
    # Row 2, Col 0: Energy meter | Row 2, Col 1: Stats
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)
    
    ax_wave  = fig.add_subplot(gs[0, :])   # Spans all columns in row 0
    ax_fft   = fig.add_subplot(gs[1, :])   # Spans all columns in row 1
    ax_energy = fig.add_subplot(gs[2, 0])  # Bottom-left
    ax_stats  = fig.add_subplot(gs[2, 1])  # Bottom-right
    
    # ── Panel 1: Waveform ─────────────────────────────────────────────────
    plot_waveform(audio_data, sample_rate, 
                  title="🎵 Waveform (Amplitude vs Time)", ax=ax_wave)
    
    # ── Panel 2: FFT Spectrum ─────────────────────────────────────────────
    plot_fft(audio_data, sample_rate,
             title="📊 FFT Magnitude Spectrum (Frequency Content)", ax=ax_fft)
    
    # ── Panel 3: Energy Meter ──────────────────────────────────────────────
    # Use a horizontal bar chart as a "meter"
    # Color the bar based on energy level:
    #   Green = quiet (< -40 dB), Yellow = moderate, Red = loud (> -10 dB)
    
    if db > -10:
        bar_color = "#FF4444"    # Red — loud / clipping risk
        level_label = "LOUD"
    elif db > -30:
        bar_color = "#FFD700"    # Yellow — moderate speech level
        level_label = "MODERATE"
    else:
        bar_color = "#44BB44"    # Green — quiet / noise
        level_label = "QUIET"
    
    # Clamp energy display to 0-100%
    # Map dB range [-80, 0] → [0%, 100%]
    energy_pct = max(0.0, min(100.0, (db + 80) / 80 * 100))
    
    ax_energy.barh(
        ["Audio Level"],      # y-axis label
        [energy_pct],         # bar length (0-100)
        color=bar_color,
        height=0.4,
        edgecolor="black",
        linewidth=0.8,
    )
    ax_energy.set_xlim(0, 100)
    ax_energy.set_xlabel("Energy Level (%)", fontsize=10)
    ax_energy.set_title(f"⚡ Energy: {db:.1f} dB [{level_label}]", 
                        fontsize=12, fontweight="bold")
    ax_energy.set_facecolor("#F8F9FA")
    ax_energy.grid(True, axis="x", alpha=0.3)
    
    # Add threshold markers
    ax_energy.axvline(x=25, color="green", linewidth=1.5, linestyle=":", alpha=0.7)
    ax_energy.axvline(x=75, color="red",   linewidth=1.5, linestyle=":", alpha=0.7)
    ax_energy.text(25, 0.7, "Quiet", fontsize=7, color="green", ha="center")
    ax_energy.text(75, 0.7, "Loud",  fontsize=7, color="red",   ha="center")
    
    # ── Panel 4: Statistics Table ──────────────────────────────────────────
    ax_stats.axis("off")  # Hide axes — we just want text
    
    stats_text = (
        f"📋  AUDIO STATISTICS\n"
        f"{'─' * 28}\n"
        f"  Duration        : {duration:.2f} s\n"
        f"  Sample Rate     : {sample_rate:,} Hz\n"
        f"  Total Samples   : {len(audio_data):,}\n"
        f"  Channels        : Mono (1)\n"
        f"{'─' * 28}\n"
        f"  RMS Energy      : {rms:.4f}\n"
        f"  Level (dB)      : {db:.1f} dB\n"
        f"  Peak Amplitude  : {float(np.max(np.abs(audio_data)))/32767:.4f}\n"
        f"{'─' * 28}\n"
        f"  Dominant Freq   : {dom_freq:.1f} Hz\n"
        f"  FFT Resolution  : {sample_rate/len(audio_data):.2f} Hz/bin\n"
        f"  Nyquist Limit   : {sample_rate//2:,} Hz"
    )
    
    ax_stats.text(
        0.05, 0.98, stats_text,
        transform=ax_stats.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",  # Monospaced font for alignment
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#EEF2FF",
            edgecolor="#7B68EE",
            alpha=0.8,
        ),
    )
    
    # ── Save or Show ──────────────────────────────────────────────────────
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[DSP] 🖼️  Dashboard saved to: '{save_path}'")
    
    return fig


# ─── Real-Time Streaming Visualizer ──────────────────────────────────────────

class RealTimeVisualizer:
    """
    A real-time audio visualizer that updates plots as audio is recorded.

    Uses matplotlib.animation.FuncAnimation to redraw the plot every
    few milliseconds with new audio data from a shared buffer.

    Architecture:
    - A numpy array (ring buffer) holds the most recent N samples
    - sounddevice callback fills the buffer as audio arrives
    - FuncAnimation reads the buffer and redraws the plot
    - Both run in parallel — callback is on audio thread, animation on main thread

    Usage:
    ------
        viz = RealTimeVisualizer(sample_rate=44100, window_seconds=3.0)
        viz.start()
        # ... do your recording here ...
        viz.stop()
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        window_seconds: float = 3.0,   # Show 3 seconds of audio history
        update_interval_ms: int = 50,  # Redraw every 50ms (20 fps)
        max_freq_display: float = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.window_size = int(window_seconds * sample_rate)  # Buffer size in samples
        self.update_interval = update_interval_ms
        self.max_freq_display = max_freq_display
        
        # Ring buffer: holds the last `window_size` audio samples
        # Initialized to silence (zeros)
        self._buffer = np.zeros(self.window_size, dtype=np.float32)
        self._lock = threading.Lock()  # Prevent race conditions on buffer
        
        self._running = False
        self._stream = None
        self._anim = None
        self._fig = None
    
    def _audio_callback(
        self,
        indata: np.ndarray,    # New audio chunk from microphone
        frames: int,            # Number of frames in this chunk
        time_info,              # Timing info (not used here)
        status,                 # Error status
    ) -> None:
        """
        Called by sounddevice on a background thread every time new audio arrives.
        
        We roll the buffer (shift left) and insert the new chunk at the end.
        np.roll shifts all elements by -frames positions, then we overwrite 
        the last `frames` elements with the new data.
        
        This lock-free approach minimizes audio glitches:
        - We only hold the lock during the array write
        - The animation thread holds the lock only during the read
        """
        if status:
            print(f"[Visualizer] Audio callback status: {status}")
        
        # Flatten and normalize incoming chunk
        chunk = indata.flatten().astype(np.float32)
        
        with self._lock:
            # Roll buffer left by len(chunk) positions
            self._buffer = np.roll(self._buffer, -len(chunk))
            # Write new chunk at end
            self._buffer[-len(chunk):] = chunk
    
    def _setup_figure(self) -> Tuple[Figure, list]:
        """Create the figure and axes for real-time visualization."""
        fig, axes = plt.subplots(
            2, 1, figsize=(12, 6),
            gridspec_kw={"hspace": 0.4},
        )
        fig.suptitle("🎙️  Real-Time Audio Visualizer", fontsize=14, fontweight="bold")
        
        # Waveform axis (top)
        axes[0].set_title("Waveform", fontsize=11)
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_ylim(-1.1, 1.1)
        axes[0].set_facecolor("#0D1117")  # Dark background for live feel
        axes[0].tick_params(colors="white")
        
        # FFT axis (bottom)
        axes[1].set_title("FFT Spectrum", fontsize=11)
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Magnitude")
        axes[1].set_xlim(0, self.max_freq_display)
        axes[1].set_facecolor("#0D1117")
        axes[1].tick_params(colors="white")
        
        fig.patch.set_facecolor("#161B22")  # Dark figure background
        
        return fig, axes
    
    def _update_frame(self, frame_idx: int, axes: list, lines: list) -> list:
        """
        Called by FuncAnimation on every frame to redraw the plots.
        
        This function:
        1. Reads current buffer (with lock for thread safety)
        2. Computes FFT
        3. Updates the line data on each plot
        
        Updating line data (line.set_data()) is faster than clearing 
        and re-plotting — crucial for smooth real-time animation.
        """
        with self._lock:
            current_buffer = self._buffer.copy()  # Thread-safe copy
        
        # Time axis for waveform
        time_axis = np.arange(len(current_buffer)) / self.sample_rate
        
        # FFT
        freqs, mags = compute_fft(current_buffer, self.sample_rate)
        dom_freq, _ = get_dominant_frequency(freqs, mags)
        rms = compute_rms_energy(current_buffer)
        db  = rms_to_db(rms)
        
        # Update waveform line data
        lines[0].set_data(time_axis, current_buffer)
        axes[0].set_xlim(0, time_axis[-1])
        axes[0].set_title(
            f"Waveform  |  RMS: {rms:.4f}  |  Level: {db:.1f} dB",
            fontsize=10, color="white",
        )
        
        # Update FFT line data (filter to display range)
        mask = freqs <= self.max_freq_display
        lines[1].set_data(freqs[mask], mags[mask])
        axes[1].set_ylim(0, max(mags.max() * 1.1, 0.001))
        axes[1].set_title(
            f"FFT Spectrum  |  Dominant: {dom_freq:.1f} Hz",
            fontsize=10, color="white",
        )
        
        return lines
    
    def start(self) -> None:
        """
        Start the real-time visualizer.
        
        1. Opens microphone stream (sounddevice InputStream)
        2. Sets up matplotlib figure
        3. Creates FuncAnimation for live updates
        4. Shows the figure (non-blocking: plt.show(block=False))
        """
        import sounddevice as sd
        
        if self._running:
            print("[Visualizer] Already running!")
            return
        
        self._running = True
        
        # Open microphone input stream
        # This calls _audio_callback every time new audio is available
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
            blocksize=1024,  # Process 1024 samples per callback call
        )
        self._stream.start()
        print("[Visualizer] 📊 Real-time visualizer started. Close the window to stop.")
        
        # Setup figure and initial lines
        self._fig, axes = self._setup_figure()
        
        time_axis = np.zeros(self.window_size)
        freqs_init = np.linspace(0, self.max_freq_display, self.window_size // 2)
        
        # Create line objects — we'll update their data each frame
        wave_line, = axes[0].plot(
            time_axis, np.zeros_like(time_axis),
            color="#00FF88", linewidth=0.8, animated=True,
        )
        fft_line, = axes[1].plot(
            freqs_init, np.zeros_like(freqs_init),
            color="#00BFFF", linewidth=1.0, animated=True,
        )
        lines = [wave_line, fft_line]
        
        # FuncAnimation calls _update_frame every `interval` milliseconds
        # blit=True means only changed artists are redrawn (faster)
        self._anim = animation.FuncAnimation(
            self._fig,
            func=lambda frame: self._update_frame(frame, axes, lines),
            interval=self.update_interval,
            blit=True,
            cache_frame_data=False,
        )
        
        plt.show(block=True)  # Block until the window is closed
    
    def stop(self) -> None:
        """Stop the real-time visualizer and close audio stream."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            plt.close(self._fig)
        print("[Visualizer] Stopped.")


# ─── AI-Adaptive DSP Additions ────────────────────────────────────────────────

def estimate_snr(audio: np.ndarray, sample_rate: int) -> float:
    """
    Estimate the Signal-to-Noise Ratio (SNR) in dB.
    Uses the first 0.5 seconds of the audio signal as a noise reference.
    
    SNR (dB) = 20 * log10(RMS(Signal) / RMS(Noise))
    """
    # Normalize to float32
    if audio.dtype == np.int16:
        audio_float = audio.astype(np.float32) / 32767.0
    else:
        audio_float = audio.astype(np.float32)
        
    noise_len = int(0.5 * sample_rate)
    
    # If audio is shorter than 0.5s, use the first 10% as noise
    if len(audio_float) <= noise_len:
        noise_len = max(1, len(audio_float) // 10)
        
    noise_samples = audio_float[:noise_len]
    signal_samples = audio_float[noise_len:]
    
    if len(signal_samples) == 0:
        signal_samples = audio_float # fallback
        
    rms_noise = compute_rms_energy(noise_samples)
    rms_signal = compute_rms_energy(signal_samples)
    
    if rms_noise == 0.0:
        return 50.0  # High SNR if noise is silent
        
    snr_db = 20.0 * np.log10(rms_signal / (rms_noise + 1e-10))
    return float(snr_db)


def noise_spectrum(audio: np.ndarray, sample_rate: int, n_fft: int = 1024) -> dict:
    """
    Evaluate the FFT of the noise reference (first 0.5 seconds of audio).
    Returns frequencies, magnitudes, and the noise floor level (dB).
    """
    # Normalize to float32
    if audio.dtype == np.int16:
        audio_float = audio.astype(np.float32) / 32767.0
    else:
        audio_float = audio.astype(np.float32)
        
    noise_len = int(0.5 * sample_rate)
    if len(audio_float) <= noise_len:
        noise_len = max(1, len(audio_float) // 10)
        
    noise_samples = audio_float[:noise_len]
    
    # Crop or pad to match exact n_fft size
    if len(noise_samples) < n_fft:
        noise_samples = np.pad(noise_samples, (0, n_fft - len(noise_samples)), mode="constant")
    else:
        noise_samples = noise_samples[:n_fft]
        
    freqs, mags = compute_fft(noise_samples, sample_rate)
    rms_noise = compute_rms_energy(noise_samples)
    noise_floor_db = rms_to_db(rms_noise)
    
    return {
        "freqs": freqs,
        "magnitudes": mags,
        "noise_floor": noise_floor_db
    }


def extract_feature_summary(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Extract a summary of audio features using librosa and numpy.
    Used by the AI Decision Engine to choose the best DSP pipeline path.
    
    Returns a dictionary of metrics:
    - noise_level: RMS of noise reference in dB
    - snr_db: Estimated SNR in dB
    - spectral_centroid: Center of gravity of the spectrum (Hz)
    - zero_crossing_rate: Ratio of sign changes in signal
    - bandwidth: Width of the spectrum (Hz)
    - dominant_frequency: Frequency of peak magnitude (Hz)
    """
    try:
        import librosa
    except ImportError:
        raise ImportError(
            "[DSP] librosa is not installed but required for feature extraction. "
            "Please check requirements.txt."
        )
        
    # Normalize to float32
    if audio.dtype == np.int16:
        audio_float = audio.astype(np.float32) / 32767.0
    else:
        audio_float = audio.astype(np.float32)
        
    # 1. Noise level and SNR
    noise_samples = audio_float[:int(0.5 * sample_rate)]
    rms_noise = compute_rms_energy(noise_samples)
    noise_level = rms_to_db(rms_noise)
    snr_db = estimate_snr(audio_float, sample_rate)
    
    # 1b. Overall signal RMS (full signal, for silence detection)
    rms_overall = compute_rms_energy(audio_float)
    rms_db_overall = rms_to_db(rms_overall)
    
    # 2. Spectral Features using librosa
    # We take the mean across all frames for global summary
    spectral_centroid_frames = librosa.feature.spectral_centroid(y=audio_float, sr=sample_rate)[0]
    spectral_centroid = float(np.mean(spectral_centroid_frames))
    
    zero_crossing_rate_frames = librosa.feature.zero_crossing_rate(y=audio_float)[0]
    zero_crossing_rate = float(np.mean(zero_crossing_rate_frames))
    
    bandwidth_frames = librosa.feature.spectral_bandwidth(y=audio_float, sr=sample_rate)[0]
    bandwidth = float(np.mean(bandwidth_frames))
    
    # 3. Dominant frequency using existing FFT code
    freqs, mags = compute_fft(audio_float, sample_rate)
    dom_freq, _ = get_dominant_frequency(freqs, mags)
    dominant_frequency = float(dom_freq)
    
    return {
        "noise_level": noise_level,
        "snr_db": snr_db,
        "spectral_centroid": spectral_centroid,
        "zero_crossing_rate": zero_crossing_rate,
        "bandwidth": bandwidth,
        "dominant_frequency": dominant_frequency,
        "rms_db_overall": rms_db_overall,
        "sample_rate": sample_rate
    }

