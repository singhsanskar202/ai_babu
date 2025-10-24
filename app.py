import streamlit as st
from openai import OpenAI
import tempfile
import os
from pydub import AudioSegment
from speech_recognition import Recognizer, AudioFile

# ------------------------------
# SETUP
# ------------------------------
st.set_page_config(page_title="AI Business Consultant", page_icon="💼", layout="centered")

# Load API key from Streamlit secrets
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

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
"""

# ------------------------------
# FUNCTIONS
# ------------------------------
def transcribe_audio(audio_bytes):
    """Convert uploaded audio bytes to text using SpeechRecognition."""
    recognizer = Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        sound = AudioSegment.from_file(tmp.name)
        sound.export(tmp.name, format="wav")

        with AudioFile(tmp.name) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
    return text


def get_consultant_reply(user_text):
    """Query OpenRouter model for business advice."""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",
        messages=[
            {"role": "system", "content": CONSULTANT_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=0.6,
        max_tokens=350,
    )
    return response.choices[0].message.content.strip()


def speak_text(text):
    """Render JavaScript-based speech synthesis."""
    js = f"""
    <script>
    const utterance = new SpeechSynthesisUtterance({text!r});
    utterance.pitch = 1;
    utterance.rate = 1.05;
    utterance.volume = 1;
    speechSynthesis.speak(utterance);
    </script>
    """
    st.components.v1.html(js, height=0, width=0)


# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("💼 McKinsey-Style Talking Consultant")
st.markdown("Speak your business challenge — your AI consultant will listen, analyze, and respond like a strategy expert.")

audio_input = st.audio_input("🎙 Speak your question clearly")

if st.button("💬 Ask Consultant"):
    if audio_input is not None:
        with st.spinner("🎧 Processing your voice..."):
            try:
                user_text = transcribe_audio(audio_input.read())
                st.markdown(f"**👤 You said:** {user_text}")

                reply = get_consultant_reply(user_text)
                st.markdown(f"**💼 Consultant:** {reply}")

                # Speak response aloud
                speak_text(reply)

            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("Please record your voice question first.")

st.caption("Powered by OpenRouter GPT-OSS-20B | Designed as a voice-interactive business consultant.")
