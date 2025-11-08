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
    st.error("🚨 OPENROUTER_API_KEY not found. Please add it in Streamlit secrets.")
    st.stop()

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ------------------------------
# BASE PROMPTS
# ------------------------------
BASE_PROMPT = """
You are a friendly local business coach for Indian small business owners — especially from Tier 2 and Tier 3 cities.
You speak in {language}, with a natural, helpful tone.
You can use Hindi or Hinglish words (like “samjha”, “thoda dhyaan do”, “yeh common problem hai”) if the user prefers Hinglish.

You help shop owners, tutors, salon owners, kirana stores, traders, and small restaurant owners with:
sales, marketing, customer retention, pricing, and day-to-day business problems.

Your advice should be:
- Simple and realistic for Indian conditions.
- Cost-effective (focus on WhatsApp, word-of-mouth, or simple promotions).
- Include real-world examples (like kirana, salon, coaching class, etc.).
- Avoid jargon or corporate words.

Always follow this structure:
1️⃣ Start with empathy (e.g. “Samjha, yeh common issue hai.”)
2️⃣ Give 2–3 simple, actionable tips in bullet points.
3️⃣ End with one short question to keep the conversation going.
"""

# ------------------------------
# UTILITIES
# ------------------------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("small")

def transcribe_audio(audio_bytes):
    """Convert speech to text using Whisper"""
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

def get_consultant_reply(messages, language):
    """Generate response from GPT model"""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        reply = response.choices[0].message.content.strip()

        # Add emoji formatting and spacing for clarity
        reply = reply.replace("•", "• ").replace("1.", "1️⃣").replace("2.", "2️⃣").replace("3.", "3️⃣")
        return reply
    except Exception as e:
        st.error(f"❌ Model error: {e}")
        return None

def speak_text(text, voice):
    """Generate audio for the consultant’s reply"""
    try:
        response = client.audio.speech.create(
            model="openai/tts-1",
            voice=voice,
            input=text
        )
        return response.read()
    except Exception as e:
        st.error(f"❌ TTS generation error: {e}")
        return None

def calculate_profit(cost_price, selling_price):
    """Calculate margin and provide smart advice"""
    try:
        cost = float(cost_price)
        sell = float(selling_price)
        profit = sell - cost
        margin = (profit / cost) * 100 if cost > 0 else 0
        advice = ""

        if margin < 10:
            advice = "Your margin is quite low. Try sourcing cheaper raw material or slightly increasing price."
        elif 10 <= margin < 25:
            advice = "Decent margin — you can improve it with bundle offers or loyalty discounts."
        else:
            advice = "Great margin! Focus on retaining customers and scaling sales."

        return profit, margin, advice
    except:
        return None, None, "Please enter valid numbers."

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("🇮🇳 Desi Business Coach")
st.markdown("Helping small Indian businesses — from kirana stores to coaching classes — grow with simple, practical advice.")

# Language and Voice Selectors
col_lang, col_voice = st.columns(2)
with col_lang:
    language = st.selectbox("🗣 Choose your preferred language:", ["Hinglish", "English", "Hindi"])
with col_voice:
    voice = st.selectbox("🎧 Choose voice:", ["alloy", "verse", "nova"])

# Generate language-specific system prompt
system_prompt = BASE_PROMPT.format(language=language)

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "Namaste 🙏 Main aapka business coach hoon. Batao, kis problem mein help chahiye — sales, customers, ya profit?"}
    ]

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "audio" in msg:
                st.audio(msg["audio"], format="audio/mp3")

# Microphone input
st.divider()
st.markdown("🎙 Speak your question or business problem:")
audio_bytes = mic_recorder(
    start_prompt="🎤 Start Talking",
    stop_prompt="⏹ Stop",
    just_once=True,
    key="desi_mic"
)

# ------------------------------
# HANDLE AUDIO QUERY
# ------------------------------
if audio_bytes:
    if isinstance(audio_bytes, dict) and "bytes" in audio_bytes:
        audio_data = audio_bytes["bytes"]
    else:
        audio_data = audio_bytes

    with st.spinner("🎧 Transcribing your voice..."):
        user_text = transcribe_audio(audio_data)

    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(f"_{user_text}_")

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("💼 Thinking..."):
                reply = get_consultant_reply(st.session_state.messages, language)
                st.markdown(reply)

            # Speak back
            with st.spinner("🔊 Speaking..."):
                reply_audio = speak_text(reply, voice)
                if reply_audio:
                    st.audio(reply_audio, format="audio/mp3")
                    st.session_state.messages.append({"role": "assistant", "content": reply, "audio": reply_audio})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# ------------------------------
# PROFIT CALCULATOR
# ------------------------------
st.divider()
st.subheader("🧾 Quick Profit Calculator (for Shop Owners)")
col1, col2 = st.columns(2)
with col1:
    cost_price = st.text_input("Enter your cost price (₹):")
with col2:
    selling_price = st.text_input("Enter your selling price (₹):")

if st.button("Calculate Profit 💰"):
    profit, margin, advice = calculate_profit(cost_price, selling_price)
    if profit is not None:
        st.success(f"**Profit:** ₹{profit:.2f}  |  **Margin:** {margin:.2f}%")
        st.info(advice)
    else:
        st.error(advice)

st.caption("🚀 Made for India's small business heroes • Powered by OpenRouter")
