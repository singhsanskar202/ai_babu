import streamlit as st
import speech_recognition as sr
import pyttsx3
from openai import OpenAI
import os, time
import sounddevice as sd
import numpy as np
import soundfile as sf


# ------------------------------
# SETUP
# ------------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY ,
)

recognizer = sr.Recognizer()

def get_engine():
    """Initialize and reuse a single pyttsx3 engine safely."""
    if "engine" not in st.session_state:
        st.session_state["engine"] = pyttsx3.init()
        st.session_state["engine"].setProperty("rate", 165)
        st.session_state["engine"].setProperty("volume", 0.9)
        # pick a deeper / male voice if available
        voices = st.session_state["engine"].getProperty("voices")
        for v in voices:
            if "male" in v.name.lower() or "david" in v.name.lower():
                st.session_state["engine"].setProperty("voice", v.id)
                break
    return st.session_state["engine"]

def speak(text):
    """Speak the consultant's reply safely (prevents loop errors)."""
    engine = get_engine()
    try:
        engine.say(text)
        engine.runAndWait()
    except RuntimeError:
        engine.stop()
        engine.say(text)
        engine.runAndWait()

# ------------------------------
# CONSULTANT PERSONA
# ------------------------------
CONSULTANT_PROMPT = """
You are a senior McKinsey-style management consultant speaking with a CEO client.
You communicate like a real human consultant — structured, calm, and authoritative, yet empathetic.

Guidelines:
- Confirm understanding and ask one clarifying question first.
- Then share 2–3 concise insights or recommendations.
- Use frameworks (MECE, 3Cs, 7S, Porter's Five Forces, growth levers) only when relevant.
- Never lecture or write essays. Speak in short paragraphs, like in a real conversation.
- End each response with a question or next step.

Tone:
Professional, thoughtful, collaborative — like a trusted strategy partner.
"""

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def listen_once():
    """Capture user speech and transcribe."""
    with sr.Microphone() as source:
        st.info("🎙 Listening... Speak naturally, as if in a business meeting.")
        audio = recognizer.listen(source, phrase_time_limit=8)
    try:
        text = recognizer.recognize_google(audio)
        st.success(f"👤 You said: {text}")
        return text
    except sr.UnknownValueError:
        st.warning("Sorry, I couldn’t understand that.")
        return None

def consultant_reply(history):
    """Generate a concise, realistic consultant response."""
    response = client.chat.completions.create(
        model="meta-llama/llama-4-maverick:free",
        messages=history,
        temperature=0.6,
        max_tokens=220,
        extra_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "McKinsey Consultant App",
        },
    )
    text = response.choices[0].message.content.strip()

    # Trim overly long answers (first ~5 sentences)
    parts = text.split(". ")
    trimmed = ". ".join(parts[:5]).strip()
    if not trimmed.endswith("."):
        trimmed += "."
    return trimmed

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.set_page_config(page_title="McKinsey AI Consultant (GPT-4o)", page_icon="💼", layout="centered")
st.title("💼 McKinsey-Style Strategy Consultant")
st.markdown("_A realistic, interactive voice-based consulting experience._")

if "chat" not in st.session_state:
    st.session_state["chat"] = [{"role": "system", "content": CONSULTANT_PROMPT}]

col1, col2 = st.columns(2)
with col1:
    start_button = st.button("🎤 Start Conversation")
with col2:
    stop_button = st.button("🛑 End Session")

st.divider()

# ------------------------------
# CONVERSATION LOOP
# ------------------------------
if start_button:
    st.write("🧠 Consultant ready. Speak freely — say 'stop' or 'thank you' to end.")
    running = True
    while running:
        user_input = listen_once()
        if not user_input:
            continue

        if user_input.lower() in ["stop", "thank you", "bye", "exit", "goodbye"]:
            speak("It was great speaking with you. Let's reconnect once you have the next round of data.")
            st.write("💬 Conversation ended.")
            break

        # Add user input to conversation
        st.session_state["chat"].append({"role": "user", "content": user_input})

        # Generate consultant's reply
        with st.spinner("Consultant analyzing your case..."):
            reply = consultant_reply(st.session_state["chat"])

        # Add reply and show it
        st.session_state["chat"].append({"role": "assistant", "content": reply})
        st.markdown(f"**💼 Consultant:** {reply}")
        speak(reply)
        time.sleep(1.5)  # small pause before listening again

# ------------------------------
# DISPLAY CHAT HISTORY
# ------------------------------
for m in st.session_state["chat"]:
    if m["role"] == "user":
        st.markdown(f"**👤 You:** {m['content']}")
    elif m["role"] == "assistant":
        st.markdown(f"**💼 Consultant:** {m['content']}")
