import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings, WebRtcMode
from transformers import pipeline
from vosk import Model, KaldiRecognizer
import wave
from pydub import AudioSegment
from io import BytesIO
import numpy as np
from datetime import datetime
import logging
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AI response pipeline (Hugging Face DialoGPT)
conversation_pipeline = pipeline("conversational", model="microsoft/DialoGPT-small")

# Vosk ASR model setup (download a Vosk model and specify its path)
vosk_model = Model("path_to_vosk_model")

# Text-to-Speech function using Mozilla TTS
def tts_speak(text):
    try:
        subprocess.run(["tts", "--text", text, "--out_path", "response.wav"])
        audio_file = AudioSegment.from_wav("response.wav")
        audio_bytes = BytesIO()
        audio_file.export(audio_bytes, format="wav")
        return audio_bytes.getvalue()
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        logger.error(f"TTS Error: {str(e)}")
        return None

# Function to get response from DialoGPT model
def get_ai_response(text):
    response = conversation_pipeline(text)
    return response[0]["generated_text"]

# WebRTC Audio Processor for Vosk Speech Recognition
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        self.recognizer = KaldiRecognizer(vosk_model, 16000)
    
    def recv(self, frame):
        try:
            audio_data = frame.to_ndarray()
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32768).astype(np.int16)

            wav_bytes = BytesIO(audio_data.tobytes())
            wav_bytes.seek(0)
            
            with wave.open(wav_bytes, "rb") as wf:
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if self.recognizer.AcceptWaveform(data):
                        text = self.recognizer.Result()
                        if text:
                            text = text.get("text", "")
                            if "last_text" not in st.session_state or st.session_state["last_text"] != text:
                                st.session_state["last_text"] = text
                                st.session_state["user_text"] = text

                                if "chat_history" not in st.session_state:
                                    st.session_state["chat_history"] = []

                                timestamp = datetime.now().strftime("%H:%M:%S")
                                st.session_state["chat_history"].append({
                                    "role": "user",
                                    "content": text,
                                    "timestamp": timestamp
                                })

                                ai_response = get_ai_response(text)
                                st.session_state["chat_history"].append({
                                    "role": "assistant",
                                    "content": ai_response,
                                    "timestamp": timestamp
                                })

                                st.session_state["pending_tts"] = ai_response
                                logger.info(f"Processed speech: {text}")

            return frame

        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}")
            return frame

def display_chat_history():
    if "chat_history" in st.session_state and st.session_state["chat_history"]:
        for message in st.session_state["chat_history"]:
            if message["role"] == "user":
                st.markdown(f'🗣️ **You** ({message["timestamp"]}): {message["content"]}')
            else:
                st.markdown(f'🤖 **AI** ({message["timestamp"]}): {message["content"]}')

def main():
    try:
        st.set_page_config(page_title="Voice Chat AI", layout="wide")
        st.title("Real-Time Voice Chat with AI")
        st.write("Click 'START' and begin speaking to interact with the AI!")

        if "user_text" not in st.session_state:
            st.session_state["user_text"] = ""
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        if "pending_tts" not in st.session_state:
            st.session_state["pending_tts"] = None

        col1, col2 = st.columns([2, 1])
        
        with col1:
            chat_container = st.container()
            with chat_container:
                display_chat_history()

        with col2:
            webrtc_ctx = webrtc_streamer(
                key="voice",
                mode=WebRtcMode.SENDONLY,
                audio_processor_factory=AudioProcessor,
                client_settings=ClientSettings(
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={
                        "audio": True,
                        "video": False,
                    },
                ),
            )

            if webrtc_ctx.state.playing:
                st.success("🎤 Listening...")
            else:
                st.warning("🔇 Microphone inactive")

            if st.button("Clear Chat History"):
                st.session_state["chat_history"] = []
                st.session_state["user_text"] = ""
                st.session_state["last_text"] = ""
                st.rerun()

        if st.session_state.get("pending_tts"):
            try:
                response_text = st.session_state["pending_tts"]
                audio_response = tts_speak(response_text)
                if audio_response:
                    st.audio(audio_response, format="audio/wav")
                st.session_state["pending_tts"] = None
            except Exception as e:
                logger.error(f"TTS Error: {str(e)}")

    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        logger.error(f"Main application error: {str(e)}")

if __name__ == "__main__":
    main()
