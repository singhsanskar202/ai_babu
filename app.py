import streamlit as st
from openai import OpenAI
import os
import io
# NEW IMPORT for the microphone component
from streamlit_mic_recorder import mic_recorder # <-- FIX: Renamed 'audio_recorder' to 'mic_recorder'

# ------------------------------
# SETUP
# ------------------------------
st.set_page_config(page_title="AI Business Consultant", page_icon="💼", layout="centered")

# Load API key from Streamlit secrets
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found. Please set it in your Streamlit secrets.")
    st.stop()

# Initialize the OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

CONSULTANT_PROMPT = """
You are a friendly, practical, and encouraging business coach.
Your user is a small business owner (like a cafe owner, a local plumber, or a freelancer), NOT a CEO of a big company.
Your goal is to provide simple, actionable, and easy-to-understand advice.

**YOUR PROCESS:**
1.  When the user states a problem (e.g., "sales are down"), first, show you understand and be supportive.
2.  Then, ask **ONE** simple clarifying question to get more context.
3.  Finally, give **ONE or TWO** very simple, practical, "do-it-this-week" suggestions.

**IMPORTANT RULES:**
1.  **NO JARGON:** Do NOT use consulting jargon like "MECE," "3Cs," "Porter's Five Forces," "KPIs," or "synergy."
2.  **TONE:** Be warm, empathetic, and supportive. Talk like a helpful friend who also knows a lot about business.
3.  **NEVER list 4-5 questions.** This is overwhelming and stressful for a small business owner. Just ask one simple question at a time.

Do not start with greetings — go straight to the point.
"""

# ------------------------------
# FUNCTIONS
            model="openai/whisper-1",
            file=audio_file,
            language="en"
        )
        return transcription.text
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

def get_consultant_reply(messages):
    """Get a chat-based reply from the consultant."""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            temperature=0.6,
            max_tokens=350,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error connecting to AI model: {e}")
        return None

def speak_text(text):
    """Convert text to speech using OpenRouter's TTS API."""
    try:
        response = client.audio.speech.create(
            model="openai/tts-1",
            voice="alloy",
            input=text
        )
        return response.read()
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

# ------------------------------
# STREAMLIT UI (Voice-Only)
# ------------------------------

st.title("💼 AI Business Consultant")
st.markdown("Your AI consultant is ready. Press the mic, ask your question, and stop recording.")

# 1. Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": CONSULTANT_PROMPT},
        {"role": "assistant", "content": "How can I help you frame your business challenge today?"}
    ]

# 2. Display existing chat history (This part is crucial)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # If there's audio, play it
            if "audio" in msg:
                st.audio(msg["audio"], format="audio/mp3")

# 3. REMOVED: st.chat_input (as requested)
# We are now a voice-only app.

# 4. NEW: Handle Voice Input with streamlit_mic_recorder
# This will place a mic button at the bottom of the chat
st.divider()

# We use columns to center the button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # This is the new microphone button.
    # It returns the audio bytes when the user stops recording.
    # `pause_threshold=2.5` will auto-stop recording after 2.5s of silence.
    audio_bytes = mic_recorder( # <-- FIX: Renamed 'audio_recorder' to 'mic_recorder'
        start_prompt="🎙️ Ask Consultant", # <-- FIX: Changed 'text' to 'start_prompt'
        stop_prompt="⏹️ Stop Recording",  # <-- NEW: Added a clear stop prompt
        icon_size="2.5rem",
        pause_threshold=2.5,
        sample_rate=16_000 # Use a lower sample rate for faster transcription
    )

# 5. Process the audio if we received it
if audio_bytes:
    with st.spinner("🎧 Transcribing your voice..."):
        user_text = transcribe_audio(audio_bytes)

    if user_text:
        # Add user message to state and UI
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(f"*{user_text}*") # Italicize transcribed text

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("💼 Consultant is thinking..."):
                reply = get_consultant_reply(st.session_state.messages)
                st.markdown(reply)
            
            with st.spinner("Generating audio..."):
                reply_audio_bytes = speak_text(reply)
                if reply_audio_bytes:
                    st.audio(reply_audio_bytes, format="audio/mp3")
                    # Store the audio in session state
                    st.session_state.messages.append({"role": "assistant", "content": reply, "audio": reply_audio_bytes})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": reply})
        
        # Rerun to clear the mic button and show the new messages
        st.rerun()

st.caption("Powered by OpenRouter | A voice-first business consultant.")
