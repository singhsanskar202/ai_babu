import streamlit as st
import speech_recognition as sr
from openai import OpenAI
from pydub import AudioSegment
import os
import tempfile

# ------------------------------
# SETUP
# ------------------------------
st.set_page_config(page_title="McKinsey AI Consultant", page_icon="💼", layout="centered")

# Load OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

CONSULTANT_PROMPT = """
You are a McKinsey-style senior business consultant advising a CEO.
Be clear, structured, and insightful.
Use frameworks like MECE, 3Cs, Porter's Five Forces, and 7S when relevant.
Ask clarifying questions before giving recommendations.
End with 2–3 actionable next steps.
"""

# ------------------------------
# TRANSCRIPTION (FIXED VERSION)
# ------------------------------
def transcribe_audio(temp_audio_path):
    """Convert WebM to WAV (16kHz PCM) and transcribe using Google Speech Recognition."""
    try:
        wav_path = temp_audio_path.replace(".wav", "_converted.wav")
        sound = AudioSegment.from_file(temp_audio_path)
        sound = sound.set_frame_rate(16000).set_channels(1)
        sound.export(wav_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        st.warning(f"Transcription failed: {e}")
        return ""

# ------------------------------
# CONSULTANT REPLY
# ------------------------------
def consultant_reply(history):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b:free",
            messages=history,
            temperature=0.6,
            max_tokens=300,
            extra_headers={
                "HTTP-Referer": "https://share.streamlit.io",
                "X-Title": "McKinsey Consultant App",
            },
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Model error: {e}")
        return "I'm sorry, but I encountered an issue generating a response."

# ------------------------------
# TEXT-TO-SPEECH (browser audio)
# ------------------------------
def speak_text(text):
    try:
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text
        )
        audio_bytes = speech.read()
        st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.warning(f"TTS error: {e}")

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("💼 McKinsey-Style AI Business Consultant")
st.markdown("_Speak your question, and get a consultant-grade answer._")

if "chat" not in st.session_state:
    st.session_state["chat"] = [{"role": "system", "content": CONSULTANT_PROMPT}]

audio_input = st.audio_input("🎙 Speak your business question")

if audio_input:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_input.read())
        temp_audio_path = temp_audio.name

    st.info("🎧 Processing your voice input...")

    user_text = transcribe_audio(temp_audio_path)
    if not user_text:
        user_text = "Could not understand your question."

    st.markdown(f"**👤 You:** {user_text}")

    # Consultant reply
    st.session_state["chat"].append({"role": "user", "content": user_text})
    with st.spinner("💼 Consultant thinking..."):
        reply = consultant_reply(st.session_state["chat"])

    st.session_state["chat"].append({"role": "assistant", "content": reply})
    st.markdown(f"**💼 Consultant:** {reply}")
    speak_text(reply)

# --- Chat history display ---
st.divider()
for m in st.session_state["chat"]:
    if m["role"] == "user":
        st.markdown(f"**👤 You:** {m['content']}")
    elif m["role"] == "assistant":
        st.markdown(f"**💼 Consultant:** {m['content']}")
