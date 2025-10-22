import streamlit as st
import pyttsx3
from openai import OpenAI
import os, tempfile

# ------------------------------
# SETUP
# ------------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def get_engine():
    """Initialize and reuse pyttsx3 safely"""
    if "engine" not in st.session_state:
        st.session_state["engine"] = pyttsx3.init()
        st.session_state["engine"].setProperty("rate", 165)
        st.session_state["engine"].setProperty("volume", 0.9)
    return st.session_state["engine"]

def speak(text):
    """Speak output safely."""
    engine = get_engine()
    try:
        engine.say(text)
        engine.runAndWait()
    except RuntimeError:
        engine.stop()
        engine.say(text)
        engine.runAndWait()

CONSULTANT_PROMPT = """
You are a senior McKinsey-style management consultant speaking with a CEO client.
Speak in short, structured sentences. 
Ask clarifying questions before giving advice.
Use frameworks (MECE, 3Cs, 7S, Porter's Five Forces) when relevant.
Always end with a next step or reflection question.
"""

def consultant_reply(history):
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=history,
        temperature=0.6,
        max_tokens=220,
        extra_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "McKinsey Consultant App",
        },
    )
    return response.choices[0].message.content.strip()

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.set_page_config(page_title="McKinsey AI Consultant", page_icon="💼", layout="centered")
st.title("💼 McKinsey-Style Strategy Consultant")
st.markdown("_Browser-based voice conversation demo (no PortAudio needed)._")

if "chat" not in st.session_state:
    st.session_state["chat"] = [{"role": "system", "content": CONSULTANT_PROMPT}]

# --- Record user voice in browser ---
audio_input = st.audio_input("🎙 Speak your business question")

if audio_input:
    # Save the uploaded audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_input.read())
        temp_audio_path = temp_audio.name

    st.success("🎧 Got your question. Processing...")

    # If you want to transcribe speech to text using OpenAI Whisper or another API:
    try:
        transcription = client.audio.transcriptions.create(
            model="openai/whisper-1",
            file=open(temp_audio_path, "rb"),
        )
        user_text = transcription.text.strip()
        st.markdown(f"**👤 You:** {user_text}")
    except Exception:
        user_text = "Transcription failed."
        st.warning(user_text)

    # Get consultant reply
    st.session_state["chat"].append({"role": "user", "content": user_text})
    with st.spinner("Consultant thinking..."):
        reply = consultant_reply(st.session_state["chat"])

    st.session_state["chat"].append({"role": "assistant", "content": reply})
    st.markdown(f"**💼 Consultant:** {reply}")
    speak(reply)

# --- Chat history display ---
st.divider()
for m in st.session_state["chat"]:
    if m["role"] == "user":
        st.markdown(f"**👤 You:** {m['content']}")
    elif m["role"] == "assistant":
        st.markdown(f"**💼 Consultant:** {m['content']}")
