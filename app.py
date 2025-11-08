import streamlit as st
from openai import OpenAI
import os
import io
from streamlit_mic_recorder import mic_recorder  # ✅ Correct import

# ------------------------------
# SETUP
# ------------------------------
st.set_page_config(page_title="AI Business Consultant", page_icon="💼", layout="centered")

# Load API key
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("🚨 OPENROUTER_API_KEY not found. Please set it in your Streamlit secrets.")
    st.stop()

# Initialize the OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

CONSULTANT_PROMPT = """
You are a McKinsey-style senior business consultant talking to a CEO.
You are strategic, structured, and confident.
Ask clarifying questions before making recommendations.
Use frameworks like MECE, 3Cs, Porter's Five Forces, and 7S when relevant.
Provide concise, actionable insights, in a natural conversational tone.
Do not start your first message with a greeting — go straight to the point.
"""

# ------------------------------
# FUNCTIONS
# ------------------------------

def transcribe_audio(audio_bytes):
    """Transcribe speech using Whisper via OpenRouter."""
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "input_audio.wav"

        transcription = client.audio.transcriptions.create(
            model="openai/whisper-1",
            file=audio_file,
            language="en"
        )
        return transcription.text
    except Exception as e:
        st.error(f"❌ Transcription error: {e}")
        return None


def get_consultant_reply(messages):
    """Generate a business consulting response."""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            temperature=0.6,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"❌ Error connecting to model: {e}")
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
        st.error(f"❌ Text-to-speech error: {e}")
        return None


# ------------------------------
# STREAMLIT APP
# ------------------------------
st.title("💼 AI Business Consultant")
st.markdown("Speak your business challenge — your AI consultant will listen, analyze, and talk back.")

# Initialize chat state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": CONSULTANT_PROMPT},
        {"role": "assistant", "content": "How can I help you frame your business challenge today?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "audio" in msg:
                st.audio(msg["audio"], format="audio/mp3")

# Divider and mic control
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    audio_bytes = mic_recorder(
        start_prompt="🎙️ Start Speaking",
        stop_prompt="⏹️ Stop Recording",
        pause_threshold=2.5,
        sample_rate=16000
    )

# Process new voice input
if audio_bytes:
    with st.spinner("🎧 Transcribing your voice..."):
        user_text = transcribe_audio(audio_bytes)

    if user_text:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(f"_{user_text}_")

        # Get consultant’s reply
        with st.chat_message("assistant"):
            with st.spinner("💼 Consultant is thinking..."):
                reply = get_consultant_reply(st.session_state.messages)
                st.markdown(reply)

            # Generate speech
            with st.spinner("🎙️ Generating audio response..."):
                reply_audio = speak_text(reply)
                if reply_audio:
                    st.audio(reply_audio, format="audio/mp3")
                    st.session_state.messages.append({"role": "assistant", "content": reply, "audio": reply_audio})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": reply})

        # Refresh chat to reset mic UI
        st.rerun()

st.caption("Powered by OpenRouter • Voice-first McKinsey-style AI Consultant")
