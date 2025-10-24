import streamlit as st
from openai import OpenAI
import tempfile
import os
from pydub import AudioSegment
from speech_recognition import Recognizer, AudioFile
import speech_recognition

# ------------------------------
# SETUP
# ------------------------------
st.set_page_config(page_title="AI Business Consultant", page_icon="💼", layout="centered")

# Load API key from Streamlit secrets
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

# Check if API key is loaded
if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found. Please set it in your Streamlit secrets.")
    st.stop()

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
def transcribe_audio(audio_file_data):
    """Convert uploaded audio data to text using SpeechRecognition."""
    recognizer = Recognizer()

    # 1. Load the audio data directly from the in-memory file-like object
    try:
        # REMOVED format="ogg" to let ffmpeg auto-detect the format
        sound = AudioSegment.from_file(audio_file_data)
        
        # *** NEW STEP: Boost audio volume ***
        # Boost the audio by 10dB. This can help if the recording is too quiet.
        sound = sound + 10

    except Exception as e:
        st.error(f"Error loading audio with pydub: {e}. Please try recording again.")
        return None

    # 2. Export this sound to a WAV format in a temp file
    #    This is what speech_recognition.AudioFile needs.
    wav_filename = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            sound.export(tmp_wav.name, format="wav")
            wav_filename = tmp_wav.name

        # 3. Transcribe the WAV file
        with AudioFile(wav_filename) as source:
            # *** NEW STEP: Adjust for ambient noise ***
            # Listen for 0.5 seconds to adjust for noise
            try:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
            except Exception as e:
                st.warning(f"Could not adjust for ambient noise: {e}")
                
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        return text
        
    except speech_recognition.UnknownValueError:
        st.warning("Google Speech Recognition could not understand the audio. Please try speaking more clearly.")
        return None
    except speech_recognition.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")
        return None
    finally:
        # 4. Clean up the temp file
        if wav_filename and os.path.exists(wav_filename):
            os.remove(wav_filename)


def get_consultant_reply(user_text):
    """Query OpenRouter model for business advice."""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini", # Using a more standard, capable model
            messages=[
                {"role": "system", "content": CONSULTANT_PROMPT},
                {"role": "user", "content": user_text},
            ],
            temperature=0.6,
            max_tokens=350,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error connecting to AI model: {e}")
        return None


def speak_text(text):
    """Render JavaScript-based speech synthesis."""
    # Escape backticks, newlines, and other special characters for JS
    safe_text = text.replace('`', '\\`').replace('\n', '\\n').replace("'", "\\'")
    
    js = f"""
    <script>
        try {{
            const utterance = new SpeechSynthesisUtterance('{safe_text}');
            utterance.pitch = 1;
            utterance.rate = 1.05;
            utterance.volume = 1;
            
            // Log for debugging
            console.log("Attempting to speak: {safe_text.split('\\n')[0]}...");

            // Cancel any previous speech to avoid overlap
            speechSynthesis.cancel();
            speechSynthesis.speak(utterance);
        }} catch (e) {{
            console.error("Speech synthesis error:", e);
        }}
    </script>
    """
    st.components.v1.html(js, height=0, width=0)


# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("💼 AI Business Consultant")
st.markdown("Speak your business challenge — your AI consultant will listen, analyze, and respond like a strategy expert.")

audio_input = st.audio_input("🎙 Speak your question clearly")

if st.button("💬 Ask Consultant"):
    if audio_input is not None:
        with st.spinner("🎧 Processing your voice..."):
            # Pass the audio_input object directly, NOT audio_input.read()
            user_text = transcribe_audio(audio_input) 

            if user_text:
                st.markdown(f"**👤 You said:** {user_text}")
                
                with st.spinner("💼 Consultant is thinking..."):
                    reply = get_consultant_reply(user_text)
                
                if reply:
                    st.markdown(f"**💼 Consultant:** {reply}")
                    # Speak response aloud
                    speak_text(reply)

    else:
        st.warning("Please record your voice question first.")

st.caption("Powered by OpenRouter | Designed as a voice-interactive business consultant.")

