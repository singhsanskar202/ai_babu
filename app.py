import streamlit as st
import speech_recognition as sr
from openai import OpenAI
import os
import tempfile

# ------------------------------
# SETUP
# ------------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

CONSULTANT_PROMPT = """
You are a seasoned McKinsey-style management consultant speaking to a CEO.
Speak clearly and confidently.
Ask clarifying questions before giving recommendations.
Use frameworks like MECE, 3Cs, Porter's Five Forces, or 7S when relevant.
Conclude with 2–3 actionable next steps or strategic options.
"""

# ------------------------------
# TRANSCRIPTION HELPER
# ------------------------------
def transcribe_audio(temp_audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.warning("Speech not understood.")
        return ""
    except Exception as e:
        st.warning(f"Transcription failed: {e}")
        return ""

# ------------------------------
# CONSULTANT REPLY
# ------------------------------
def consultant_reply(history):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",
        messages=history,
        temperature=0.6,
        max_tokens=220,
        extra_headers={
            "HTTP-Referer": "https://share.streamlit.io",
            "X-Title": "McKinsey Consultant App",
        },
    )
    return response.choices[0].message.content.strip()

# ------------------------------
# TEXT TO SPEECH (browser-safe)
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
st.set_page_config(page_title="McKinsey AI Consultant", page_icon="💼", layout="centered")
st.title("💼 McKinsey-Style Strategy Consultant")
st.markdown("_Ask your AI business advisor about growth, operations, or strategy._")

if "chat" not in st.session_state:
    st.session_state["chat"] = [{"role": "system", "content": CONSULTANT_PROMPT}]

# --- Record user voice in browser ---
audio_input = st.audio_input("🎙 Speak your business question")

if audio_input:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_input.read())
        temp_audio_path = temp_audio.name

    # Transcribe audio using Google SpeechRecognition
    user_text = transcribe_audio(temp_audio_path)
    if not user_text:
        user_text = "Could not understand your question."
    st.markdown(f"**👤 You:** {user_text}")

    # Get consultant reply
    st.session_state["chat"].append({"role": "user", "content": user_text})
    with st.spinner("Consultant thinking..."):
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


