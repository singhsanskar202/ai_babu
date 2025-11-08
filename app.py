import streamlit as st
from openai import OpenAI
import os
import tempfile
import whisper
from streamlit_mic_recorder import mic_recorder

# ------------------------------
# SETUP
# ------------------------------
st.set_page_config(page_title="🇮🇳 Desi Business Coach", page_icon="💼", layout="centered")

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
You are a friendly Indian business coach who helps small business owners — especially from Tier 2 and Tier 3 cities.
You speak in *Hinglish* — a mix of simple English and easy Hindi words.
You give advice that feels local, real, and practical for Indian shopkeepers, freelancers, and small entrepreneurs.

Your style:
- Talk like a friend or mentor, not a corporate consultant.
- Use short, clear sentences.
- Avoid big English words or frameworks.
- Use examples from Indian life — kirana stores, salons, tuition classes, restaurants, clothing shops, etc.
- Focus on local, low-cost ideas (like WhatsApp promotions, word-of-mouth, small loyalty offers).

Your structure:
1️⃣ Start with empathy (e.g., “Samjha, yeh common problem hai.”)
2️⃣ Give 2–3 simple, actionable tips — focus on what they can do today.
3️⃣ End with one short, friendly question to keep the chat going.

Always sound warm, encouraging, and practical.
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
    """Generate a Hinglish, desi-style practical reply"""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=450,
        )
        reply = response.choices[0].message.content.strip()

        # Polish readability and add emoji touch
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
            voice="alloy",  # Indian-friendly English voice
            input=text
        )
        return response.read()
    except Exception as e:
        st.error(f"❌ TTS generation error: {e}")
        return None

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("🇮🇳 Desi Business Coach")
st.markdown("Talk about your business problem — sales, customers, ya daily struggles — and I’ll give you simple, practical advice that works in India.")

# Initialize chat memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": CONSULTANT_PROMPT},
        {"role": "assistant", "content": "Namaste 🙏 Main aapka business coach hoon. Batao, kis problem mein help chahiye — sales, customers, ya profit?"}
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

    with st.spinner("🎧 Sun raha hoon... samajhne ki koshish kar raha hoon..."):
        user_text = transcribe_audio(audio_data)

    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(f"🗣️ *{user_text}*")

        # Generate consultant reply
        with st.chat_message("assistant"):
            with st.spinner("💼 Soch raha hoon..."):
                reply = get_consultant_reply(st.session_state.messages)
                st.markdown(reply)

            # Speak reply
            with st.spinner("🎙️ Bol raha hoon..."):
                reply_audio = speak_text(reply)
                if reply_audio:
                    st.audio(reply_audio, format="audio/mp3")
                    st.session_state.messages.append({"role": "assistant", "content": reply, "audio": reply_audio})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": reply})

        st.rerun()

st.caption("💬 Powered by OpenRouter • Made for India's small business heroes 🇮🇳")
