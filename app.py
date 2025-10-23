import streamlit as st
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
You are a senior McKinsey-style management consultant talking to a CEO.
Be structured, calm, and clear.
Ask clarifying questions first.
Use frameworks like MECE, 3Cs, Porter's Five Forces, and 7S when relevant.
Always end with 2-3 actionable recommendations or next steps.
"""

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def consultant_reply(history):
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=history,
        temperature=0.6,
        max_tokens=220,
        extra_headers={
            "HTTP-Referer": "https://share.streamlit.io",
            "X-Title": "McKinsey Consultant App",
        },
    )
    return response.choices[0].message.content.strip()


def speak_text(text):
    """Generate and play back AI voice using OpenAI TTS (browser-compatible)."""
    try:
        speech = client.audio.speech.create(
            model="openai/gpt-oss-20b:free",
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
st.markdown("_Voice-based business advisor — works in Streamlit Cloud._")

if "chat" not in st.session_state:
    st.session_state["chat"] = [{"role": "system", "content": CONSULTANT_PROMPT}]

audio_input = st.audio_input("🎙 Speak your business question")

if audio_input:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_input.read())
        temp_audio_path = temp_audio.name

    # Transcribe user question
    try:
        transcription = client.audio.transcriptions.create(
            model="openai/whisper-1",
            file=open(temp_audio_path, "rb"),
        )
        user_text = transcription.text.strip()
    except Exception:
        user_text = "Transcription failed."

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

