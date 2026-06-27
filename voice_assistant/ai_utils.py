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

# System prompt that defines the AI's persona and behavior
DEFAULT_SYSTEM_PROMPT: str = """You are a helpful, friendly, and conversational AI voice assistant.

Important rules for voice responses:
- Keep responses concise and natural for spoken audio (2-4 sentences max unless asked for more)
- Avoid using markdown, bullet points, or formatting — speak in plain conversational prose
- Use simple, clear language that sounds natural when spoken aloud
- If you don't know something, say so clearly and briefly
- You can have memory of the current conversation

Your personality: warm, helpful, knowledgeable, and slightly witty."""


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
        
        # The message list — start empty (system prompt added separately)
        self._messages: List[Dict[str, str]] = []
    
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


# ─── OpenAI Backend ───────────────────────────────────────────────────────────

def _query_openai(
    user_text: str,
    history: ConversationHistory,
    model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = 0.7,   # 0 = deterministic, 1 = creative
    max_tokens: int = 1024,     # Limit response length
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
    
    # Get API key from environment variable (never hardcode keys!)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "[AI] OPENAI_API_KEY environment variable not set.\n"
            "     Set it with: $env:OPENAI_API_KEY = 'sk-...'\n"
            "     Get a key at: https://platform.openai.com/api-keys"
        )
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Add user message to history before sending
    history.add_user_message(user_text)
    messages = history.get_openai_messages()
    
    print(f"[AI] 🤖 Querying OpenAI ({model})...")
    start_time = time.time()
    
    try:
        # Make the API call
        response = client.chat.completions.create(
            model=model,
            messages=messages,          # Full conversation history
            temperature=temperature,    # Response randomness
            max_tokens=max_tokens,      # Response length limit
        )
        
        elapsed = time.time() - start_time
        
        # Extract the response text
        # response.choices is a list (usually 1 choice)
        ai_text = response.choices[0].message.content.strip()
        
        # Log usage info (helpful for monitoring costs)
        usage = response.usage
        print(f"[AI] ✅ Response received in {elapsed:.2f}s")
        print(f"[AI]    Tokens used: {usage.prompt_tokens} prompt + "
              f"{usage.completion_tokens} completion = {usage.total_tokens} total")
        
        # Add AI response to history for next turn
        history.add_assistant_message(ai_text)
        
        return ai_text
    
    except Exception as e:
        # Remove the user message we added (since we didn't get a response)
        history._messages.pop()
        print(f"[AI] ❌ OpenAI API error: {e}")
        raise


# ─── Google Gemini Backend ────────────────────────────────────────────────────

def _query_gemini(
    user_text: str,
    history: ConversationHistory,
    model: str = DEFAULT_GEMINI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
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
    
    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "[AI] GEMINI_API_KEY environment variable not set.\n"
            "     Set it with: $env:GEMINI_API_KEY = 'AIza...'\n"
            "     Get a free key at: https://aistudio.google.com/app/apikey"
        )
    
    # Configure the API key
    genai.configure(api_key=api_key)
    
    # Add user message to history
    history.add_user_message(user_text)
    
    print(f"[AI] 🤖 Querying Gemini ({model})...")
    start_time = time.time()
    
    try:
        # Initialize the model with generation config
        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=history.system_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        
        # Build chat history in Gemini format (exclude the last user message
        # since we'll send it via send_message)
        chat_history = []
        for msg in history._messages[:-1]:  # All except last (current user msg)
            role = "model" if msg["role"] == "assistant" else "user"
            chat_history.append({
                "role": role,
                "parts": [msg["content"]]
            })
        
        # Start a chat session with history
        chat = gemini_model.start_chat(history=chat_history)
        
        # Send the current message
        response = chat.send_message(user_text)
        
        elapsed = time.time() - start_time
        ai_text = response.text.strip()
        
        print(f"[AI] ✅ Response received in {elapsed:.2f}s")
        
        # Add AI response to history
        history.add_assistant_message(ai_text)
        
        return ai_text
    
    except Exception as e:
        # Remove the user message we added
        history._messages.pop()
        print(f"[AI] ❌ Gemini API error: {e}")
        raise


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


# ─── Main Query Function (Public API) ─────────────────────────────────────────

def query_ai(
    user_text: str,
    history: ConversationHistory,
    backend: str = DEFAULT_BACKEND,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
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
    
    print(f"\n[AI] 📤 User said: \"{user_text}\"")
    print(f"[AI]    Backend: {backend.upper()}")
    print(f"[AI]    Conversation turn: {history.turn_count + 1}")
    
    backend = backend.lower().strip()
    
    if backend == "openai":
        chosen_model = model or DEFAULT_OPENAI_MODEL
        return _query_openai(user_text, history, chosen_model, temperature, max_tokens)
    
    elif backend in ("gemini", "google"):
        chosen_model = model or DEFAULT_GEMINI_MODEL
        return _query_gemini(user_text, history, chosen_model, temperature, max_tokens)
    
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
