import streamlit as st
from openai import OpenAI
import os
import io
import tempfile
import whisper
from streamlit_mic_recorder import mic_recorder

# ------------------------------
# SETUP
# ------------------------------
st.set_page_config(page_title="🧑‍💼 Friendly Business Coach", page_icon="💼", layout="centered")

# Load API key
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("🚨 OPENROUTER_API_KEY not found. Please add it in your Streamlit secrets.")
    st.stop()

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ------------------------------
# CONSULTANT PERSONALITY
# ------------------------------
CONSULTANT_PROMPT = """
You are a friendly and experienced small business advisor.
You help shop owners, freelancers, and small teams solve real business problems — like sales, customer service, pricing, and marketing.

Your tone:
- Warm, simple, and encouraging — no corporate jargon.
- Speak like a trusted mentor, not a professor or consultant.
- Use short sentences. Avoid frameworks or fancy terms.
- Focus on what the user can actually do next.

Your response style:
- Always start with empathy (“Got it”, “I understand”, “That makes sense”).
- Then give 2–3 clear, practical steps they can take.
- End with one short question to keep the conversation going.
"""

# ------------------------------
# FUNCTIONS
# ------------------------------
@st.cache_resource
def load_whisper():
    """Load Whisper model once (small = faster)"""
    return whisper.load_model("small")

def transcribe_audio(audio_bytes):
    """Transcribe using local Whisper model"""
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
    """Generate an empathetic, practical reply"""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            temperature=0.6,
            max_tokens=400,
        )
        reply = response.choices[0].message.content.strip()

        # Simple readability polish
        reply = reply.replace("•", "• ").replace("1.", "1️⃣").replace("2.", "2️⃣").replace("3.", "3️⃣")
        return reply
    except Exception as e:
        st.error(f"❌ Error connecting to AI model: {e}")
        return None

def speak_text(text):
    """Generate consultant’s voice reply"""
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
# STREAMLIT UI
# ------------------------------
st.title("🧑‍💼 Friendly Business Coach")
st.markdown("Ask about your sales, marketing, or daily business struggles — I’ll give you simple, practical advice you can act on right away.")

# Initialize chat memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": CONSULTANT_PROMPT},
        {"role": "assistant", "content": "Hey there 👋 What part of your business are you struggling with right now?"}
    ]

# Display chat
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "audio" in msg:
                st.audio(msg["audio"], format="audio/mp3")

# Microphone input
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    audio_bytes = mic_recorder(
        start_prompt="🎙️ Start Talking",
        stop_prompt="⏹️ Stop",
        just_once=True,
        key="business_mic"
    )

# ------------------------------
# HANDLE VOICE INPUT
# ------------------------------
if audio_bytes:
    # handle dict or bytes
    if isinstance(audio_bytes, dict) and "bytes" in audio_bytes:
        audio_data = audio_bytes["bytes"]
    else:
        audio_data = audio_bytes

    with st.spinner("🎧 Listening and understanding..."):
        user_text = transcribe_audio(audio_data)

    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(f"_{user_text}_")

        # Generate consultant reply
        with st.chat_message("assistant"):
            with st.spinner("💼 Thinking..."):
                reply = get_consultant_reply(st.session_state.messages)
                st.markdown(reply)

            # Speak reply
            with st.spinner("🎙️ Speaking..."):
                reply_audio = speak_text(reply)
                if reply_audio:
                    st.audio(reply_audio, format="audio/mp3")
                    st.session_state.messages.append({"role": "assistant", "content": reply, "audio": reply_audio})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": reply})

        st.rerun()

st.caption("💬 Powered by OpenRouter • Your simple, voice-first business advisor.")
