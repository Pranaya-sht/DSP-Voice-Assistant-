"""
ai_utils.py - AI Language Model Integration
============================================
This module handles communication with AI language model APIs.

Supports two backends (configurable):
1. OpenAI (GPT-4o, GPT-3.5-turbo, etc.)
2. Google Gemini (gemini-1.5-flash, gemini-1.5-pro, etc.)

Key concepts for beginners:
- API (Application Programming Interface): A way for programs to talk to 
  external services over the internet using HTTP requests.
- API Key: A secret string that identifies your account to the service.
  Keep it private — never share or commit to public repos!
- Conversation history: A list of previous messages (user + assistant) 
  sent with each request so the AI remembers the context.
- Streaming: Receiving the AI's response token-by-token (like typing) 
  instead of waiting for the full response. Makes it feel more responsive.
- System prompt: An initial instruction that sets the AI's persona/behavior.
"""

import os           # For reading environment variables (API keys)
import time         # For measuring response latency
from typing import List, Dict, Optional, Generator

# We import AI clients lazily (inside functions) so the app still starts 
# even if a library isn't installed. This is called "lazy importing".


# ─── Configuration ────────────────────────────────────────────────────────────

# Default AI backend: "openai" or "gemini"
# Change this to switch between providers, or set via environment variable:
#   $env:AI_BACKEND = "gemini"  (PowerShell)
#   export AI_BACKEND=gemini    (Bash)
DEFAULT_BACKEND: str = os.environ.get("AI_BACKEND", "openai")

# Default models for each backend (can be overridden via environment variables)
DEFAULT_OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")     # Cheaper, fast, very capable
DEFAULT_GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash") # Active, fast, and free tier available

# Explanation styles users can pick (UI) or ask for (voice/text)
EXPLANATION_STYLES: Dict[str, str] = {
    "auto": (
        "Pick the clearest style for each question. Use simple language by default; "
        "go deeper only when the user asks."
    ),
    "simple": (
        "Explain like I'm 10: very short sentences, everyday words, zero jargon. "
        "Use fun comparisons a child would understand."
    ),
    "college": (
        "Explain like a college student: correct terminology, logical steps, moderate depth. "
        "Define technical terms briefly when first used."
    ),
    "math": (
        "Explain mathematically: include relevant formulas, variables, and signal-processing "
        "notation (e.g., SNR in dB, Fourier transform, filter transfer functions). "
        "Still keep it speakable — read formulas aloud in words."
    ),
    "visual": (
        "Explain visually: describe what the waveform and spectrum LOOK like — peaks, valleys, "
        "hills, flat lines, bright vs dim frequency bands. Guide the listener to imagine the plot."
    ),
    "analogy": (
        "Explain with analogy: tie every concept to everyday sounds or objects "
        "(ocean waves, radio static, humming fridge, whisper vs shout, windshield wipers)."
    ),
}

EXPLANATION_STYLE_ALIASES: Dict[str, str] = {
    "eli10": "simple",
    "el10": "simple",
    "kid": "simple",
    "child": "simple",
    "beginner": "simple",
    "simple": "simple",
    "college": "college",
    "student": "college",
    "undergrad": "college",
    "math": "math",
    "mathematical": "math",
    "mathematics": "math",
    "equation": "math",
    "visual": "visual",
    "visually": "visual",
    "picture": "visual",
    "analogy": "analogy",
    "analogies": "analogy",
    "metaphor": "analogy",
    "auto": "auto",
    "normal": "auto",
    "default": "auto",
}

STYLE_TRIGGER_PHRASES: Dict[str, List[str]] = {
    "simple": [
        "explain like i'm 10", "explain like i am 10", "explain like a kid",
        "explain simply", "eli10", "like i'm five", "explain like i'm 5",
    ],
    "college": [
        "explain like a college student", "college level", "undergraduate",
    ],
    "math": [
        "explain mathematically", "with math", "with equations", "show the formula",
    ],
    "visual": [
        "explain visually", "paint a picture", "describe what it looks like",
    ],
    "analogy": [
        "explain with analogy", "explain with an analogy", "use an analogy",
        "compare it to something",
    ],
}


def normalize_explanation_style(style: Optional[str]) -> str:
    if not style:
        return "auto"
    key = style.lower().strip().replace("-", "_").replace(" ", "_")
    return EXPLANATION_STYLE_ALIASES.get(key, key if key in EXPLANATION_STYLES else "auto")


def detect_explanation_style_from_text(text: str) -> Optional[str]:
    """Detect if the user is asking for a specific explanation style."""
    lower = text.lower()
    for style, phrases in STYLE_TRIGGER_PHRASES.items():
        if any(p in lower for p in phrases):
            return style
    return None


def style_display_name(style: str) -> str:
    names = {
        "auto": "Auto",
        "simple": "Explain like I'm 10",
        "college": "College student",
        "math": "Mathematical",
        "visual": "Visual",
        "analogy": "Analogy",
    }
    return names.get(normalize_explanation_style(style), style)


BASE_SYSTEM_PROMPT: str = """You are a voice-first AI assistant that is BOTH a friendly general chatbot AND a digital signal processing (DSP) expert.

Your dual role:
1. General assistant — answer everyday questions naturally (weather, facts, advice, small talk).
2. DSP & audio expert — explain waveforms, spectra, filters, noise, speech, and the analysis data provided. Suggest real-world applications when helpful.

Real-world DSP applications you can mention when relevant:
- Hearing aids and cochlear implants (bandpass, noise reduction)
- Phone and video calls (echo cancellation, noise suppression)
- Smart speakers and voice assistants (VAD, beamforming)
- Music production and podcasts (EQ, compression, de-noising)
- Medical ultrasound and sonar (frequency analysis)
- Radar and wireless communications (filtering, SNR)
- Audio forensics and security (environment classification)

How input works:
- [DSP ANALYSIS CONTEXT] blocks are computed BEFORE you respond (preprocessing, features, Random Forest filter choice, adaptive DSP, CNN14, SHAP). Treat them as ground truth for that audio turn.
- [USER MESSAGE] is what the person said or typed. Answer their intent using conversation history plus DSP context when relevant.
- On follow-up messages without new audio, use the most recent DSP context and prior chat — do not pretend you re-analyzed audio.

Conversation rules:
- Maintain multi-turn context: refer back to earlier questions naturally.
- If the user switches topics, follow them; if they return to audio, reconnect to the last analysis.
- When they ask "what applications" or "where is this used", give 2–4 concrete examples tied to their audio metrics when possible.
- If speech was not detected or context is missing, say so honestly.

Voice output rules (text-to-speech):
- Write natural spoken prose. No markdown, bullet lists, or code blocks unless the user explicitly asks for technical detail.
- For simple questions: 2–4 sentences. For teaching moments: up to 8–10 sentences.
- Read numbers clearly (e.g., "fifteen decibels" or "15 dB" — both fine for TTS).

Personality: warm, patient, encouraging — like a favorite teacher who loves audio engineering."""


def build_system_prompt(explanation_style: str = "auto") -> str:
    """Build the full system prompt with the active explanation style."""
    style = normalize_explanation_style(explanation_style)
    style_text = EXPLANATION_STYLES.get(style, EXPLANATION_STYLES["auto"])
    return (
        f"{BASE_SYSTEM_PROMPT}\n\n"
        f"ACTIVE EXPLANATION STYLE ({style.upper()}):\n{style_text}\n\n"
        "If the user asks to change explanation style, adapt immediately on the next reply."
    )


DEFAULT_SYSTEM_PROMPT: str = build_system_prompt("auto")


# ─── Conversation History Manager ────────────────────────────────────────────

class ConversationHistory:
    """
    Manages the conversation history for multi-turn dialogue.

    The history is a list of message dictionaries:
    [
        {"role": "user",      "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user",      "content": "What's 2+2?"},
        ...
    ]

    This is sent with every API call so the model has full context.
    OpenAI and Gemini both use this same format (with minor differences).
    """
    
    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_history_turns: int = 10,  # Keep last N turns to limit token usage
    ):
        """
        Parameters:
        -----------
        system_prompt     : str - The AI's instructions/persona
        max_history_turns : int - Max conversation turns to keep in memory
                                  (1 turn = 1 user message + 1 assistant reply)
                                  Older turns are dropped to save API tokens.
        """
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.explanation_style: str = "auto"
        
        # The message list — start empty (system prompt added separately)
        self._messages: List[Dict[str, str]] = []

    def set_explanation_style(self, style: str) -> str:
        """Update explanation style and refresh the system prompt."""
        self.explanation_style = normalize_explanation_style(style)
        self.system_prompt = build_system_prompt(self.explanation_style)
        return self.explanation_style

    def get_ui_messages(self) -> List[Dict[str, str]]:
        """Return conversation messages for the web UI."""
        return [{"role": m["role"], "content": m["content"]} for m in self._messages]
    
    def add_user_message(self, text: str) -> None:
        """Add a user message to the history."""
        self._messages.append({"role": "user", "content": text})
        self._trim()
    
    def add_assistant_message(self, text: str) -> None:
        """Add an assistant response to the history."""
        self._messages.append({"role": "assistant", "content": text})
    
    def _trim(self) -> None:
        """
        Remove oldest messages if history exceeds max_history_turns.
        
        Each "turn" is 2 messages (user + assistant).
        We always keep an even number of messages to maintain pairs.
        """
        max_messages = self.max_history_turns * 2
        if len(self._messages) > max_messages:
            # Drop the oldest messages (from the front of the list)
            self._messages = self._messages[-max_messages:]
    
    def get_openai_messages(self) -> List[Dict[str, str]]:
        """
        Get messages formatted for OpenAI API.
        
        OpenAI format: system message first, then alternating user/assistant.
        """
        return [
            {"role": "system", "content": self.system_prompt}
        ] + self._messages
    
    def get_gemini_messages(self) -> List[Dict]:
        """
        Get messages formatted for Google Gemini API.
        
        Gemini uses "model" instead of "assistant" for AI responses.
        Gemini also doesn't have a "system" role in the same way —
        we prepend the system prompt to the first user message instead.
        """
        if not self._messages:
            return []
        
        gemini_messages = []
        for i, msg in enumerate(self._messages):
            role = "model" if msg["role"] == "assistant" else "user"
            content = msg["content"]
            
            # Prepend system prompt to the first user message
            if i == 0 and msg["role"] == "user":
                content = f"{self.system_prompt}\n\nUser: {content}"
            
            gemini_messages.append({
                "role": role,
                "parts": [{"text": content}]
            })
        
        return gemini_messages
    
    def clear(self) -> None:
        """Clear all conversation history (but keep the system prompt)."""
        self._messages = []
        print("[AI] Conversation history cleared.")
    
    def display_history(self) -> None:
        """Print the conversation history for debugging."""
        print("\n[AI] 💬 Conversation History:")
        print("─" * 50)
        for i, msg in enumerate(self._messages):
            role = msg["role"].upper()
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            print(f"  [{i}] {role}: {content}")
        print("─" * 50)
    
    @property
    def turn_count(self) -> int:
        """Number of complete turns (user + assistant pairs)."""
        return len(self._messages) // 2


# ─── API Key Helpers (multi-key fallback) ────────────────────────────────────

def get_gemini_api_keys() -> List[str]:
    """Collect Gemini API keys from env (supports comma-separated list and numbered keys)."""
    keys: List[str] = []
    multi = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GOOGLE_API_KEYS")
    if multi:
        keys.extend(k.strip() for k in multi.split(",") if k.strip())
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"):
        val = os.environ.get(name)
        if val and val not in keys:
            keys.append(val.strip())
    return keys


def get_openai_api_keys() -> List[str]:
    """Collect OpenAI API keys from env."""
    keys: List[str] = []
    multi = os.environ.get("OPENAI_API_KEYS")
    if multi:
        keys.extend(k.strip() for k in multi.split(",") if k.strip())
    for name in ("OPENAI_API_KEY", "OPENAI_API_KEY_2", "OPENAI_API_KEY_3"):
        val = os.environ.get(name)
        if val and val not in keys:
            keys.append(val.strip())
    return keys


# ─── OpenAI Backend ───────────────────────────────────────────────────────────

def _query_openai(
    user_text: str,
    history: ConversationHistory,
    model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = 0.7,   # 0 = deterministic, 1 = creative
    max_tokens: int = 1024,     # Limit response length
    dsp_context: Optional[str] = None,
    is_followup: bool = False,
) -> str:
    """
    Send a query to the OpenAI API and get a response.

    Parameters:
    -----------
    user_text   : str - The user's transcribed speech
    history     : ConversationHistory - Current conversation context
    model       : str - Which OpenAI model to use
    temperature : float - Response randomness (0.0-2.0)
    max_tokens  : int - Maximum response length in tokens (~4 chars/token)

    Returns:
    --------
    str - The AI's text response

    OpenAI API structure:
    - client.chat.completions.create() sends messages to the chat endpoint
    - response.choices[0].message.content extracts the text response
    - Usage tokens tell us how many tokens were consumed (affects cost)
    """
    try:
        # Lazy import — only load if using this backend
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "[AI] OpenAI library not installed.\n"
            "     Run: pip install openai"
        )
    
    # Get API keys (try multiple for reliability)
    api_keys = get_openai_api_keys()
    if not api_keys:
        raise ValueError(
            "[AI] OPENAI_API_KEY environment variable not set.\n"
            "     Set it with: $env:OPENAI_API_KEY = 'sk-...'\n"
            "     Or multiple: $env:OPENAI_API_KEYS = 'sk-1,sk-2'\n"
            "     Get a key at: https://platform.openai.com/api-keys"
        )
    
    # Store plain transcript in history; enrich only for this API call
    history.add_user_message(user_text)
    messages = history.get_openai_messages()
    if dsp_context:
        messages[-1]["content"] = _compose_user_message(user_text, dsp_context, is_followup=is_followup)
    
    print(f"[AI] 🤖 Querying OpenAI ({model})...")
    start_time = time.time()
    last_error = None

    for i, api_key in enumerate(api_keys):
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed = time.time() - start_time
            ai_text = response.choices[0].message.content.strip()
            usage = response.usage
            key_label = f"key #{i + 1}" if len(api_keys) > 1 else "primary key"
            print(f"[AI] ✅ Response received in {elapsed:.2f}s ({key_label})")
            print(f"[AI]    Tokens used: {usage.prompt_tokens} prompt + "
                  f"{usage.completion_tokens} completion = {usage.total_tokens} total")
            history.add_assistant_message(ai_text)
            return ai_text
        except Exception as e:
            last_error = e
            print(f"[AI] ⚠️ OpenAI key #{i + 1} failed: {e}")
            if i < len(api_keys) - 1:
                time.sleep(0.5)

    history._messages.pop()
    print(f"[AI] ❌ OpenAI API error (all keys exhausted): {last_error}")
    raise last_error


# ─── Google Gemini Backend ────────────────────────────────────────────────────

def _query_gemini(
    user_text: str,
    history: ConversationHistory,
    model: str = DEFAULT_GEMINI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    dsp_context: Optional[str] = None,
    is_followup: bool = False,
) -> str:
    """
    Send a query to Google's Gemini API and get a response.

    Parameters:
    -----------
    user_text   : str - The user's transcribed speech
    history     : ConversationHistory - Current conversation context
    model       : str - Which Gemini model to use
    temperature : float - Response randomness
    max_tokens  : int - Maximum output tokens

    Returns:
    --------
    str - The AI's text response

    Gemini API structure:
    - Uses google.generativeai package
    - GenerativeModel.start_chat() creates a chat session
    - chat.send_message() sends a message and gets a response
    - response.text extracts the text content
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "[AI] Google Generative AI library not installed.\n"
            "     Run: pip install google-generativeai"
        )
    
    # Get API keys (try multiple for reliability)
    api_keys = get_gemini_api_keys()
    if not api_keys:
        raise ValueError(
            "[AI] GEMINI_API_KEY environment variable not set.\n"
            "     Set it with: $env:GEMINI_API_KEY = 'AIza...'\n"
            "     Or multiple: $env:GEMINI_API_KEYS = 'key1,key2,key3'\n"
            "     Get a free key at: https://aistudio.google.com/app/apikey"
        )
    
    # Store plain transcript in history; enrich only for this API call
    history.add_user_message(user_text)
    message_to_send = _compose_user_message(user_text, dsp_context, is_followup=is_followup)
    
    print(f"[AI] 🤖 Querying Gemini ({model})...")
    start_time = time.time()
    last_error = None

    chat_history = []
    for msg in history._messages[:-1]:
        role = "model" if msg["role"] == "assistant" else "user"
        chat_history.append({"role": role, "parts": [msg["content"]]})

    for i, api_key in enumerate(api_keys):
        try:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(
                model_name=model,
                system_instruction=history.system_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            chat = gemini_model.start_chat(history=chat_history)
            response = chat.send_message(message_to_send)
            elapsed = time.time() - start_time
            ai_text = response.text.strip()
            key_label = f"key #{i + 1}" if len(api_keys) > 1 else "primary key"
            print(f"[AI] ✅ Response received in {elapsed:.2f}s ({key_label})")
            history.add_assistant_message(ai_text)
            return ai_text
        except Exception as e:
            last_error = e
            print(f"[AI] ⚠️ Gemini key #{i + 1} failed: {e}")
            if i < len(api_keys) - 1:
                time.sleep(0.5)

    history._messages.pop()
    print(f"[AI] ❌ Gemini API error (all keys exhausted): {last_error}")
    raise last_error


# ─── Wake Word Detection ──────────────────────────────────────────────────────

def check_wake_word(
    text: str,
    wake_words: Optional[List[str]] = None,
) -> bool:
    """
    Check if the transcribed text contains a wake word.

    Wake words trigger the assistant to listen for a command.
    This is a simple keyword-based approach — more advanced systems 
    use dedicated models like Picovoice Porcupine.

    Parameters:
    -----------
    text       : str - The transcribed speech text
    wake_words : List[str] - Words/phrases that trigger the assistant
                 Default: ["hello assistant", "hey assistant", "ok assistant",
                           "assistant", "wake up"]

    Returns:
    --------
    bool - True if a wake word was detected, False otherwise
    
    For a more robust wake word, the text is:
    - Lowercased (case-insensitive matching)
    - Stripped of leading/trailing whitespace
    - Checked for substring containment (not exact match)
    """
    if wake_words is None:
        wake_words = [
            "hello assistant",
            "hey assistant",
            "ok assistant",
            "okay assistant",
            "assistant",
            "wake up",
        ]
    
    # Normalize: lowercase and strip whitespace
    text_lower = text.lower().strip()
    
    for wake_word in wake_words:
        # Check if the wake word appears anywhere in the text
        if wake_word.lower() in text_lower:
            print(f"[AI] 🔔 Wake word detected: '{wake_word}'")
            return True
    
    return False


# ─── DSP Context Builder ─────────────────────────────────────────────────────

def build_dsp_context(
    turn: int,
    pipeline_info: dict,
    feature_summary: dict,
    cnn14_predictions: Optional[List[Dict]] = None,
    freq_importance: Optional[Dict] = None,
    waveform_stats: Optional[Dict] = None,
    rf_importances: Optional[Dict] = None,
) -> str:
    """
    Build a structured text summary of DSP pipeline results for the LLM.
    All analysis modules run before this is sent to the AI.
    """
    lines = [
        f"[DSP ANALYSIS CONTEXT — Turn {turn}]",
        "The following was computed from the user's audio BEFORE this message:",
        "",
        "Pipeline decision:",
        f"  Filter applied     : {pipeline_info.get('filter', 'none').upper()}",
        f"  FFT size           : {pipeline_info.get('fft_size', '—')} samples",
        f"  Feature set        : {pipeline_info.get('feature_set', '—').upper()}",
        f"  Decision rule      : {pipeline_info.get('rule', 'random_forest')}",
        f"  Processing time    : {pipeline_info.get('elapsed_ms', 0):.1f} ms",
        "",
        "Acoustic features (from waveform analysis):",
        f"  SNR                : {feature_summary.get('snr_db', 0):.2f} dB",
        f"  Noise floor        : {feature_summary.get('noise_level', feature_summary.get('noise_level_db', 0)):.2f} dB",
        f"  Zero crossing rate : {feature_summary.get('zero_crossing_rate', 0):.4f}",
        f"  Dominant frequency : {feature_summary.get('dominant_frequency', feature_summary.get('dominant_freq_hz', 0)):.1f} Hz",
        f"  Spectral centroid  : {feature_summary.get('spectral_centroid', feature_summary.get('spectral_centroid_hz', 0)):.1f} Hz",
        f"  Bandwidth          : {feature_summary.get('bandwidth', feature_summary.get('bandwidth_hz', 0)):.1f} Hz",
    ]

    if waveform_stats:
        lines.extend([
            "",
            "Waveform summary:",
            f"  Duration           : {waveform_stats.get('duration_s', 0):.2f} s",
            f"  Sample rate        : {waveform_stats.get('sample_rate_hz', 0)} Hz",
            f"  Raw peak amplitude : {waveform_stats.get('raw_peak', 0):.4f}",
            f"  Filtered peak      : {waveform_stats.get('filtered_peak', 0):.4f}",
            f"  Raw RMS level      : {waveform_stats.get('raw_rms', 0):.4f}",
            f"  Filtered RMS level : {waveform_stats.get('filtered_rms', 0):.4f}",
        ])
        if waveform_stats.get("peak_reduction_pct") is not None:
            lines.append(
                f"  Peak change (filt) : {waveform_stats['peak_reduction_pct']:+.1f}% vs raw"
            )

    if rf_importances:
        top_rf = sorted(rf_importances.items(), key=lambda x: x[1], reverse=True)[:4]
        lines.extend(["", "Top RF feature importances (filter decision):"])
        for name, weight in top_rf:
            lines.append(f"  {name.replace('_', ' ')}: {weight:.3f}")

    if cnn14_predictions:
        lines.extend(["", "CNN14 environment classification (top predictions):"])
        for pred in cnn14_predictions[:3]:
            label = pred.get("label", "Unknown")
            prob = pred.get("probability", 0)
            lines.append(f"  {label}: {prob:.1%}")

    if freq_importance:
        top_bands = freq_importance.get("top_frequencies", [])
        if top_bands:
            bands = ", ".join(f"{b['freq']:.0f} Hz" for b in top_bands[:5])
            lines.extend(["", f"SHAP top frequency bands driving filter choice: {bands}"])

    lines.extend([
        "",
        "Use this context to explain waveforms, spectra, filters, and audio quality when the user asks.",
        "When asked about applications, connect metrics to real uses (calls, hearing aids, music, speech AI, sonar, etc.).",
    ])
    return "\n".join(lines)


def build_followup_context(dsp_context: str) -> str:
    """Wrap prior DSP analysis for follow-up text chat (no new audio)."""
    return (
        "[REFERENCE — DSP ANALYSIS FROM LAST AUDIO TURN]\n"
        f"{dsp_context.strip()}\n\n"
        "This is reference data from the user's most recent recording. "
        "Use it to answer follow-up questions; do not claim you re-ran the pipeline."
    )


def _compose_user_message(user_text: str, dsp_context: Optional[str], is_followup: bool = False) -> str:
    """Combine DSP analysis context with the user's message for the LLM."""
    if not dsp_context:
        return user_text.strip()
    context_block = build_followup_context(dsp_context) if is_followup else dsp_context.strip()
    return f"{context_block}\n\n[USER MESSAGE]\n{user_text.strip()}"


# ─── Main Query Function (Public API) ─────────────────────────────────────────

def query_ai(
    user_text: str,
    history: ConversationHistory,
    backend: str = DEFAULT_BACKEND,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1536,
    dsp_context: Optional[str] = None,
    explanation_style: Optional[str] = None,
    is_followup: bool = False,
) -> str:
    """
    Send user text to the configured AI backend and return the response.
    
    This is the main function that main.py calls.
    It routes to the appropriate backend (OpenAI or Gemini) based on config.

    Parameters:
    -----------
    user_text   : str - The user's transcribed speech
    history     : ConversationHistory - Conversation context
    backend     : str - "openai" or "gemini"
    model       : str or None - Override the default model
    temperature : float - Response creativity (0=deterministic, 1=creative)
    max_tokens  : int - Max response length in tokens
    dsp_context : str or None - DSP pipeline analysis summary prepended to the user message
    explanation_style : str or None - Override explanation style (auto, simple, college, math, visual, analogy)
    is_followup : bool - True for text follow-ups using reference DSP context from last audio turn

    Returns:
    --------
    str - The AI's text response

    Raises:
    -------
    ValueError  - If the backend is invalid
    ImportError - If required library isn't installed
    """
    if not user_text.strip():
        return "I didn't catch that — could you please repeat?"

    # Auto-detect style from user message (e.g. "explain like I'm 10")
    detected = detect_explanation_style_from_text(user_text)
    if detected:
        history.set_explanation_style(detected)
    elif explanation_style:
        history.set_explanation_style(explanation_style)
    
    print(f"\n[AI] 📤 User said: \"{user_text[:120]}{'...' if len(user_text) > 120 else ''}\"")
    print(f"[AI]    Backend: {backend.upper()}")
    print(f"[AI]    Style: {style_display_name(history.explanation_style)}")
    print(f"[AI]    Conversation turn: {history.turn_count + 1}")
    if dsp_context:
        print(f"[AI]    DSP context attached ({len(dsp_context)} chars, followup={is_followup})")

    backend = backend.lower().strip()
    
    if backend == "openai":
        chosen_model = model or DEFAULT_OPENAI_MODEL
        return _query_openai(
            user_text, history, chosen_model, temperature, max_tokens,
            dsp_context, is_followup,
        )
    
    elif backend in ("gemini", "google"):
        chosen_model = model or DEFAULT_GEMINI_MODEL
        return _query_gemini(
            user_text, history, chosen_model, temperature, max_tokens,
            dsp_context, is_followup,
        )
    
    else:
        raise ValueError(
            f"[AI] Unknown backend: '{backend}'. "
            "Use 'openai' or 'gemini'."
        )


# ─── Command Handler ──────────────────────────────────────────────────────────

def handle_special_commands(text: str, history: ConversationHistory) -> Optional[str]:
    """
    Handle special voice commands before sending to the AI.

    These commands are handled locally (no API call needed):
    - "clear history" / "forget everything" → clear conversation history
    - "show history" → display conversation turns
    - "quit" / "exit" / "goodbye" → signal to shut down

    Parameters:
    -----------
    text    : str - Transcribed user speech
    history : ConversationHistory

    Returns:
    --------
    str - Response message if command was handled
    None - If not a special command (should be sent to AI)
    """
    text_lower = text.lower().strip()
    
    # Style switch commands (no API call)
    style_commands = {
        "simple": ["explain like i'm 10", "explain like i am 10", "use simple mode", "kid mode"],
        "college": ["college mode", "college student mode"],
        "math": ["math mode", "mathematical mode"],
        "visual": ["visual mode", "visual explanation mode"],
        "analogy": ["analogy mode", "use analogies"],
        "auto": ["auto mode", "normal mode", "default mode"],
    }
    for style, phrases in style_commands.items():
        if any(p in text_lower for p in phrases) and len(text_lower.split()) <= 8:
            history.set_explanation_style(style)
            return f"Got it! I'll explain things using {style_display_name(style)} style from now on."

    # Clear history commands
    if any(cmd in text_lower for cmd in ["clear history", "forget everything", "start over", "reset"]):
        history.clear()
        return "I've cleared our conversation history. Let's start fresh!"
    
    # Show history command
    if any(cmd in text_lower for cmd in ["show history", "what did we talk about"]):
        history.display_history()
        return f"I've displayed our conversation history. We've had {history.turn_count} exchanges."
    
    # Quit/exit commands
    if any(cmd in text_lower for cmd in ["quit", "exit", "goodbye", "bye", "shut down", "stop"]):
        return "__EXIT__"  # Special signal to main.py to quit the loop
    
    # Not a special command
    return None
