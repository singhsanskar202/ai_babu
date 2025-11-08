import streamlit as st
from openai import OpenAI
import os
import io
from streamlit_mic_recorder import mic_recorder  # ✅ Mic input component

# ------------------------------
# SETUP
# ------------------------------
st.set_page_config(page_title="AI Business Consultant", page_icon="💼", layout="centered")

# Load API key from Streamlit secrets or environment
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("🚨 OPENROUTER_API_KEY not found. Please add it in your Streamlit secrets.")
    st.stop()

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

CONSULTANT_PROMPT = """
You are a McKinsey-style senior business consultant talking to a CEO.
You are strategic, structured, and confident.
Ask clarifying questions before making recommendations.
Use frameworks like MECE, 3Cs, Porter's Five Forces, and 7S when relevant.
Provide concise, actionable insights in a natural conversational tone.
Do not start with greetings — go straight to the point.
"""

# ------------------------------
# FUNCTIONS
# ------------------------------
import whisper

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

def transcribe_audio(audio_bytes):
    """Transcribe using local Whisper model (no API call)."""
    try:
        model = load_whisper()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            result = model.transcribe(tmp.name)
            return result["text"].strip()
    except Exception as e:
        st.error(f"❌ Transcription error: {e}")
        return None



def get_consultant_reply(messages):
    """Generate a McKinsey-style consulting answer."""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            temperature=0.6,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"❌ Error connecting to AI model: {e}")
        return None


def speak_text(text):
    """Generate spoken audio for the consultant’s reply."""
    try:
        response = client.audio.speech.create(
            model="openai/tts-1",
            voice="alloy",
            input=text
        )
        return response.read()
    except Exception as e:
        st.error(f"❌ TTS generation error: {e}")
        return None


# ------------------------------
# STREAMLIT APP
# ------------------------------
st.title("💼 AI Business Consultant")
st.markdown("Speak your business challenge — your consultant will listen, analyze, and talk back.")

# Initialize chat memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": CONSULTANT_PROMPT},
        {"role": "assistant", "content": "How can I help you frame your business challenge today?"}
    ]

# Display past chat
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "audio" in msg:
                st.audio(msg["audio"], format="audio/mp3")

# Divider and mic section
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # ✅ Compatible mic_recorder call
    audio_bytes = mic_recorder(
        start_prompt="🎙️ Start Speaking",
        stop_prompt="⏹️ Stop Recording",
        just_once=True,
        key="consultant_mic"
    )

# ------------------------------
# HANDLE VOICE INPUT
# ------------------------------
if audio_bytes:
    # ✅ Handle both dict and bytes output
    if isinstance(audio_bytes, dict) and "bytes" in audio_bytes:
        audio_data = audio_bytes["bytes"]
    else:
        audio_data = audio_bytes

    with st.spinner("🎧 Transcribing your voice..."):
        user_text = transcribe_audio(audio_data)

    if user_text:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(f"_{user_text}_")

        # Get consultant reply
        with st.chat_message("assistant"):
            with st.spinner("💼 Consultant is thinking..."):
                reply = get_consultant_reply(st.session_state.messages)
                st.markdown(reply)

            # Speak reply
            with st.spinner("🎙️ Generating audio..."):
                reply_audio = speak_text(reply)
                if reply_audio:
                    st.audio(reply_audio, format="audio/mp3")
                    st.session_state.messages.append({"role": "assistant", "content": reply, "audio": reply_audio})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": reply})

        # Refresh app to reset mic recorder
        st.rerun()

st.caption("Powered by OpenRouter • Voice-first McKinsey-style AI Consultant")
