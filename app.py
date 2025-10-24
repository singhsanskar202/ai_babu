import streamlit as st
from openai import OpenAI
import tempfile
import os
from pydub import AudioSegment
from speech_recognition import Recognizer, AudioFile

# ------------------------------
# SETUP
# ------------------------------
st.set_page_config(page_title="AI Consultant", page_icon="💼", layout="centered")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Set in Streamlit secrets or .env
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

CONSULTANT_PROMPT = """
You are a McKinsey-style senior business consultant advising a CEO.
Be structured, analytical, and confident.
Ask clarifying questions before giving actionable recommendations.
Use frameworks like MECE, 3Cs, Porter's Five Forces, or 7S when relevant.
"""

# ------------------------------
# PROCESSING
# ------------------------------
def transcribe_audio(audio_bytes):
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
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",
        messages=[
            {"role": "system", "content": CONSULTANT_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=0.6,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("💼 McKinsey-Style AI Business Consultant")
st.markdown("Speak your question, and get a consultant-grade answer.")

audio_input = st.audio_input("🎙 Speak your business question")

if st.button("Analyze 🎧"):
    if audio_input is not None:
        with st.spinner("Processing your audio..."):
            try:
                user_text = transcribe_audio(audio_input.read())
                st.write(f"**👤 You said:** {user_text}")
                reply = get_consultant_reply(user_text)
                st.markdown(f"**💼 Consultant:** {reply}")
            except Exception as e:
                st.error(f"Error processing your input: {e}")
    else:
        st.warning("Please record your question first.")

