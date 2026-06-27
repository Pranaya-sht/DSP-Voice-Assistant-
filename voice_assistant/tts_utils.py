"""
tts_utils.py - Text-to-Speech Utilities
========================================
This module converts AI text responses into spoken audio output.

Two TTS engines are supported:
1. pyttsx3  - Offline, no internet needed, uses system voices
              Works on Windows (SAPI5), macOS (NSS), Linux (espeak)
2. edge-tts - Microsoft Edge's neural TTS, needs internet
              Much higher quality voices, many languages
              Free to use (uses Edge browser's TTS service)

Key concepts for beginners:
- TTS (Text-to-Speech): Software that converts written text to audio.
- SSML (Speech Synthesis Markup Language): XML-based format for TTS 
  control (rate, pitch, volume, pauses, emphasis).
- Async/await: Python keywords for asynchronous code — functions that 
  can "pause" and let other code run while waiting for I/O (e.g., 
  network requests, file writes) to complete.
- asyncio: Python's built-in library for writing asynchronous code.
"""

import os           # For file operations
import time         # For timing TTS operations
import asyncio      # For running async code (required by edge-tts)
import tempfile     # For creating temporary files
from typing import Optional


# ─── Configuration ────────────────────────────────────────────────────────────

# Default TTS engine: "pyttsx3" or "edge_tts"
DEFAULT_TTS_ENGINE: str = os.environ.get("TTS_ENGINE", "edge_tts")

# edge-tts voice options (Windows-style neural voices):
# Full list: run `edge-tts --list-voices` in terminal
EDGE_TTS_VOICE_EN_US_MALE:   str = "en-US-GuyNeural"      # Male, neutral
EDGE_TTS_VOICE_EN_US_FEMALE: str = "en-US-JennyNeural"    # Female, warm
EDGE_TTS_VOICE_EN_GB_MALE:   str = "en-GB-RyanNeural"     # British male
EDGE_TTS_VOICE_EN_GB_FEMALE: str = "en-GB-SoniaNeural"    # British female

DEFAULT_EDGE_VOICE: str = EDGE_TTS_VOICE_EN_US_FEMALE

# pyttsx3 defaults
DEFAULT_PYTTSX3_RATE: int  = 175   # Words per minute (default ~200)
DEFAULT_PYTTSX3_VOLUME: float = 1.0  # 0.0 to 1.0

# Audio playback
TEMP_AUDIO_DIR: str = "temp_audio"  # Folder for temporary TTS audio files


# ─── Utility Functions ────────────────────────────────────────────────────────

def ensure_temp_dir() -> None:
    """Create temp audio directory if it doesn't exist."""
    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)


def clean_text_for_speech(text: str) -> str:
    """
    Clean up text to sound better when spoken aloud.
    
    Removes or replaces elements that TTS engines handle poorly:
    - Markdown formatting (**, *, #, `, etc.)
    - URLs (say "link" instead of reading the URL)
    - Multiple spaces/newlines (compress whitespace)
    - Common abbreviations expansion
    
    Parameters:
    -----------
    text : str - Raw AI response text

    Returns:
    --------
    str - Clean text suitable for TTS
    """
    import re  # Regular expressions for pattern matching
    
    # Remove markdown bold (**text** or __text__)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    
    # Remove markdown italic (*text* or _text_)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    
    # Remove markdown headers (#, ##, ###)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove inline code (`code`)
    text = re.sub(r'`(.+?)`', r'\1', text)
    
    # Replace URLs with a placeholder
    text = re.sub(r'https?://\S+', 'a link', text)
    
    # Replace markdown bullet points with natural pauses
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    
    # Compress multiple newlines to a single space (for TTS flow)
    text = re.sub(r'\n+', ' ', text)
    
    # Compress multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


# ─── pyttsx3 Engine ──────────────────────────────────────────────────────────

class Pyttsx3Speaker:
    """
    Text-to-Speech using pyttsx3 (offline, system voices).

    pyttsx3 uses platform-specific TTS engines:
    - Windows: SAPI5 (Microsoft Speech API)
    - macOS  : NSSpeechSynthesizer
    - Linux  : espeak (install separately if missing)
    
    The pyttsx3 engine is initialized once and reused for all speech.
    Initializing it is slow (~500ms), so we do it once in __init__.
    """
    
    def __init__(
        self,
        rate: int = DEFAULT_PYTTSX3_RATE,
        volume: float = DEFAULT_PYTTSX3_VOLUME,
        voice_index: int = 0,    # 0 = first available voice, 1 = second, etc.
    ):
        """
        Initialize the pyttsx3 TTS engine.
        
        Parameters:
        -----------
        rate        : int   - Speech rate in words per minute
        volume      : float - Volume level (0.0 to 1.0)
        voice_index : int   - Which installed voice to use (0-based index)
        """
        try:
            import pyttsx3
        except ImportError:
            raise ImportError(
                "[TTS] pyttsx3 not installed.\n"
                "      Run: pip install pyttsx3"
            )
        
        print("[TTS] Initializing pyttsx3 engine...")
        
        # Initialize the engine
        self.engine = pyttsx3.init()
        
        # Set properties
        self.engine.setProperty("rate", rate)        # Words per minute
        self.engine.setProperty("volume", volume)    # 0.0 to 1.0
        
        # List available voices and select one
        voices = self.engine.getProperty("voices")
        if voices:
            print(f"[TTS]   Available voices: {len(voices)}")
            for i, v in enumerate(voices):
                print(f"[TTS]   [{i}] {v.name} ({v.languages})")
            
            # Select voice by index (with bounds check)
            idx = min(voice_index, len(voices) - 1)
            self.engine.setProperty("voice", voices[idx].id)
            print(f"[TTS]   Using voice: {voices[idx].name}")
        
        print("[TTS] pyttsx3 engine ready.")
    
    def speak(self, text: str) -> None:
        """
        Speak text using pyttsx3.
        
        engine.say() queues the text for speech.
        engine.runAndWait() processes the queue and blocks until done.
        This is synchronous — the program waits until speech finishes.
        """
        clean = clean_text_for_speech(text)
        
        if not clean:
            print("[TTS] Empty text — nothing to speak.")
            return
        
        print(f"[TTS] 🔊 Speaking (pyttsx3): \"{clean[:60]}{'...' if len(clean)>60 else ''}\"")
        
        start = time.time()
        self.engine.say(clean)
        self.engine.runAndWait()  # Blocks until speech finishes
        elapsed = time.time() - start
        
        print(f"[TTS] ✅ Speech complete ({elapsed:.1f}s)")
    
    def list_voices(self) -> None:
        """Print all available system voices."""
        voices = self.engine.getProperty("voices")
        print("\n[TTS] Available pyttsx3 voices:")
        for i, v in enumerate(voices):
            print(f"  [{i}] {v.name} | {v.languages} | {v.gender}")


# ─── edge-tts Engine ──────────────────────────────────────────────────────────

class EdgeTTSSpeaker:
    """
    High-quality Text-to-Speech using Microsoft Edge's neural TTS service.
    
    edge-tts is an open-source Python library that uses the same neural 
    TTS service powering Microsoft Edge's "Read Aloud" feature.
    
    Quality: Much better than pyttsx3 — sounds like a real person.
    Requirement: Internet connection.
    Cost: Free (no API key needed).
    
    How it works:
    1. Send text to Edge's TTS service via WebSocket
    2. Receive MP3 audio data in chunks
    3. Save to a temporary file
    4. Play the file using sounddevice or playsound
    
    Since edge-tts uses async I/O (WebSocket), we need asyncio to run it.
    """
    
    def __init__(
        self,
        voice: str = DEFAULT_EDGE_VOICE,
        rate: str = "+0%",      # Speech rate: "-20%" (slower) to "+20%" (faster)
        pitch: str = "+0Hz",    # Voice pitch: "-10Hz" (lower) to "+10Hz" (higher)
        volume: str = "+0%",    # Volume: "-10%" to "+10%"
    ):
        """
        Parameters:
        -----------
        voice  : str - Voice name (e.g., "en-US-JennyNeural")
        rate   : str - Rate adjustment (e.g., "+10%", "-5%")
        pitch  : str - Pitch adjustment (e.g., "+5Hz", "-5Hz")
        volume : str - Volume adjustment (e.g., "+0%", "+10%")
        """
        try:
            import edge_tts
            self._edge_tts = edge_tts
        except ImportError:
            raise ImportError(
                "[TTS] edge-tts not installed.\n"
                "      Run: pip install edge-tts"
            )
        
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.volume = volume
        
        ensure_temp_dir()
        print(f"[TTS] edge-tts configured: voice={voice}, rate={rate}")
    
    async def _speak_async(self, text: str, output_path: str) -> None:
        """
        Async function that generates speech and saves it to a file.
        
        'async def' defines a coroutine — a function that can be paused 
        and resumed. 'await' suspends execution until the awaited task 
        completes (e.g., waiting for data from the TTS server).
        
        edge-tts.Communicate:
        - Takes text and voice name
        - Connects to the Edge TTS service via WebSocket
        - Streams MP3 audio data back
        - save() saves the complete audio to a file
        """
        communicate = self._edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
            volume=self.volume,
        )
        # 'await' pauses here until all audio data is downloaded and saved
        await communicate.save(output_path)
    
    def speak(self, text: str) -> None:
        """
        Generate speech from text and play it through speakers.
        
        Steps:
        1. Clean the text for better speech
        2. Generate audio using edge-tts (async → run via asyncio)
        3. Play the audio file using sounddevice + scipy
        
        asyncio.run() is the bridge between synchronous and async code.
        It creates an event loop, runs the async function, and returns.
        """
        clean = clean_text_for_speech(text)
        
        if not clean:
            print("[TTS] Empty text — nothing to speak.")
            return
        
        print(f"[TTS] 🔊 Speaking (edge-tts/{self.voice}): "
              f"\"{clean[:60]}{'...' if len(clean)>60 else ''}\"")
        
        # Create a temporary MP3 file path
        # Using timestamp to avoid conflicts if called rapidly
        temp_filename = f"tts_{int(time.time()*1000)}.mp3"
        temp_path = os.path.join(TEMP_AUDIO_DIR, temp_filename)
        
        start = time.time()
        
        try:
            # Run the async TTS generation
            # asyncio.run() starts an event loop, runs the coroutine, then closes
            asyncio.run(self._speak_async(clean, temp_path))
            
            gen_time = time.time() - start
            print(f"[TTS]    Audio generated in {gen_time:.2f}s → {temp_path}")
            
            # Play the audio file
            self._play_audio_file(temp_path)
            
            elapsed = time.time() - start
            print(f"[TTS] ✅ Speech complete ({elapsed:.1f}s total)")
        
        finally:
            # Clean up temporary file even if playback fails
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass  # Ignore cleanup errors
    
    def _play_audio_file(self, filepath: str) -> None:
        """
        Play an MP3 or WAV file through system speakers.
        
        Strategy:
        1. Try sounddevice + scipy (convert MP3→WAV via pydub if needed)
        2. Fallback to playsound (simpler but less control)
        3. Fallback to os.startfile on Windows
        """
        # Try pydub + sounddevice (best quality, requires ffmpeg)
        try:
            self._play_with_sounddevice(filepath)
            return
        except Exception as e:
            print(f"[TTS]    sounddevice playback failed: {e}")
        
        # Fallback: playsound
        try:
            self._play_with_playsound(filepath)
            return
        except Exception as e:
            print(f"[TTS]    playsound playback failed: {e}")
        
        # Last resort: OS default player
        if os.name == "nt":  # Windows
            os.startfile(filepath)
            time.sleep(5)  # Give OS player time to play
        else:
            print(f"[TTS] ⚠️  Could not play audio automatically.")
            print(f"[TTS]    Manually play: {filepath}")
    
    def _play_with_sounddevice(self, filepath: str) -> None:
        """
        Play audio using sounddevice (requires pydub for MP3 decoding).
        
        MP3 is a compressed format. sounddevice needs raw PCM samples.
        pydub (with ffmpeg) converts MP3 → PCM numpy array.
        
        Steps:
        1. Load MP3 with pydub
        2. Get raw samples as numpy array
        3. Play with sounddevice.play()
        """
        import sounddevice as sd
        import numpy as np
        
        # pydub requires ffmpeg for MP3 decoding
        # Install: winget install Gyan.FFmpeg (Windows)
        from pydub import AudioSegment
        
        # Load audio file (pydub auto-detects format)
        audio_segment = AudioSegment.from_file(filepath)
        
        # Convert to numpy array
        # pydub gives raw bytes — convert to int16 array
        samples = np.array(audio_segment.get_array_of_samples())
        
        # Normalize to float32 [-1, 1]
        samples = samples.astype(np.float32) / (2**15)
        
        # Reshape for stereo if needed
        if audio_segment.channels == 2:
            samples = samples.reshape(-1, 2)
        
        sample_rate = audio_segment.frame_rate
        
        print(f"[TTS]    Playing: {sample_rate}Hz, "
              f"{audio_segment.channels}ch, "
              f"{len(samples)/sample_rate:.1f}s")
        
        # Play through speakers (blocks until done)
        sd.play(samples, samplerate=sample_rate)
        sd.wait()
    
    def _play_with_playsound(self, filepath: str) -> None:
        """Play audio using the playsound library (simpler fallback)."""
        try:
            from playsound import playsound
            playsound(filepath)
        except ImportError:
            raise ImportError(
                "[TTS] playsound not installed.\n"
                "      Run: pip install playsound==1.2.2"
            )
    
    async def list_voices_async(self) -> None:
        """List all available edge-tts voices."""
        voices = await self._edge_tts.list_voices()
        en_voices = [v for v in voices if v["Locale"].startswith("en-")]
        print(f"\n[TTS] Available edge-tts English voices ({len(en_voices)}):")
        for v in en_voices:
            print(f"  {v['ShortName']:35} | {v['Gender']:6} | {v['Locale']}")
    
    def list_voices(self) -> None:
        """Synchronous wrapper to list voices."""
        asyncio.run(self.list_voices_async())


# ─── Main speak_response Function (Public API) ────────────────────────────────

# Module-level speaker instance (created once, reused)
# None until first call — "lazy initialization" pattern
_speaker_instance: Optional[object] = None


def speak_response(
    text: str,
    engine: str = DEFAULT_TTS_ENGINE,
    voice: str = DEFAULT_EDGE_VOICE,
    rate_pyttsx3: int = DEFAULT_PYTTSX3_RATE,
) -> None:
    """
    Convert AI text response to speech and play it.
    
    This is the main function called by main.py.
    It manages a single speaker instance (reuses it across calls).
    
    Parameters:
    -----------
    text         : str - The AI response text to speak
    engine       : str - "pyttsx3" or "edge_tts"
    voice        : str - Voice name for edge-tts
    rate_pyttsx3 : int - Speech rate for pyttsx3 (words per minute)
    
    Why reuse the speaker instance?
    - pyttsx3 takes ~500ms to initialize
    - Reusing avoids that delay on every response
    - The 'global' keyword lets us modify the module-level variable
    """
    # 'global' tells Python we want to modify the module-level variable
    # Without this, _speaker_instance would be a local variable
    global _speaker_instance
    
    if not text or not text.strip():
        print("[TTS] No text to speak.")
        return
    
    # Create speaker instance on first call
    if _speaker_instance is None:
        print(f"[TTS] Initializing TTS engine: {engine}")
        
        if engine == "pyttsx3":
            _speaker_instance = Pyttsx3Speaker(rate=rate_pyttsx3)
        elif engine in ("edge_tts", "edge-tts"):
            _speaker_instance = EdgeTTSSpeaker(voice=voice)
        else:
            raise ValueError(
                f"[TTS] Unknown engine: '{engine}'. "
                "Use 'pyttsx3' or 'edge_tts'."
            )
    
    # Speak the text
    _speaker_instance.speak(text)


def reset_speaker() -> None:
    """Reset the speaker instance (useful if you want to change engine/voice)."""
    global _speaker_instance
    _speaker_instance = None
    print("[TTS] Speaker instance reset.")
