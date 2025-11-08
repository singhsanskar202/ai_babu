import streamlit as st
from openai import OpenAI
import os
import io

# ------------------------------
# SETUP
# ------------------------------
st.set_page_config(page_title="AI Business Consultant", page_icon="💼", layout="centered")

# Load API key from Streamlit secrets
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found. Please set it in your Streamlit secrets.")
    st.stop()

# Initialize the OpenRouter client
# This one client will handle Transcription, Chat, and Speech
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
Do not start your first message with a greeting. Go straight to the point.
"""

# ------------------------------
# FUNCTIONS (Now much simpler!)
# ------------------------------

def transcribe_audio(audio_bytes):
    """
    Transcribe audio using OpenRouter's Whisper API.
    This is much more reliable and simpler than the previous method.
    """
    try:
        # We need to pass a file-like object to the API
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "input_audio.wav" # Provide a dummy filename
        
        # Use OpenRouter for transcription (e.g., routing to Whisper)
        transcription = client.audio.transcriptions.create(
            model="openai/whisper-1", # You can also try other whisper versions
            file=audio_file,
            language="en" # You can specify language if needed
        )
        return transcription.text
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

def get_consultant_reply(messages):
    """
    Get a chat-based reply from the consultant.
    It now receives the *entire* history for context.
    """
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini", # Kept your model choice
            messages=messages, # Pass the whole history
            temperature=0.6,
            max_tokens=350,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error connecting to AI model: {e}")
        return None

def speak_text(text):
    """
    Convert text to speech using OpenRouter's TTS API.
    This returns audio bytes, which we can play with st.audio.
    """
    try:
        # Use OpenRouter for Text-to-Speech
        response = client.audio.speech.create(
            model="openai/tts-1", # A standard, high-quality model
            voice="alloy",       # You can try different voices: alloy, echo, fable, onyx, nova, shimmer
            input=text
        )
        # Get the raw audio bytes from the response
        return response.read()
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

# ------------------------------
# STREAMLIT UI (Stateful Chat)
# ------------------------------

st.title("💼 AI Business Consultant")
st.markdown("Speak your business challenge. Your AI consultant will listen, analyze, and respond.")

# 1. Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": CONSULTANT_PROMPT},
        {"role": "assistant", "content": "How can I help you frame your business challenge today?"}
    ]

# 2. Display existing chat history
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # If there's audio associated with an assistant message, show it
            if "audio" in msg:
                st.audio(msg["audio"], format="audio/mp3")

# 3. Use st.chat_input for text-based follow-ups (Good for accessibility)
if prompt := st.chat_input("Or, type your question..."):
    # Add user message to state and UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Consultant is thinking..."):
            reply = get_consultant_reply(st.session_state.messages)
            st.markdown(reply)
        
        with st.spinner("Generating audio..."):
            audio_bytes = speak_text(reply)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
                # Store the audio in session state
                st.session_state.messages.append({"role": "assistant", "content": reply, "audio": audio_bytes})
            else:
                st.session_state.messages.append({"role": "assistant", "content": reply})

# 4. Handle Voice Input at the bottom
st.divider()
st.markdown("### <p style='text-align: center;'>Record Your Question</p>", unsafe_allow_html=True)

# We use a button to start/stop recording
# This is a common pattern for a more controlled UX
if 'recording' not in st.session_state:
    st.session_state.recording = False

# We'll use a file uploader that *can* record
# This is a simple, built-in way to get audio
audio_input = st.file_uploader(
    "Click the microphone icon to record, or upload an audio file.", 
    type=["wav", "mp3", "m4a", "ogg"], 
    label_visibility="collapsed"
)

if audio_input and not st.session_state.get('processed_audio', False):
    st.session_state.processed_audio = True # Flag to prevent re-running
    
    with st.spinner("🎧 Transcribing your voice..."):
        audio_bytes = audio_input.read()
        user_text = transcribe_audio(audio_bytes)

    if user_text:
        # Add user message to state and UI
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(f"*{user_text}*") # Italicize transcribed text

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("💼 Consultant is thinking..."):
                reply = get_consultant_reply(st.session_state.messages)
                st.markdown(reply)
            
            with st.spinner("Generating audio..."):
                audio_bytes = speak_text(reply)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                    # Store the audio in session state
                    st.session_state.messages.append({"role": "assistant", "content": reply, "audio": audio_bytes})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": reply})
        
        # Rerun to clear the file uploader and show the new messages
        st.session_state.processed_audio = False
        st.rerun()

st.caption("Powered by OpenRouter | A stateful, voice-interactive business consultant.")
