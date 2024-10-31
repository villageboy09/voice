import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import torch
from transformers import pipeline
import edge_tts
import asyncio
from vosk import Model, KaldiRecognizer
import numpy as np
import wave
from pydub import AudioSegment
import io
from datetime import datetime
import logging
import tempfile
import os
import ffmpeg

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AI response pipeline (using a smaller model for better performance)
@st.cache_resource
def load_conversation_model():
    try:
        return pipeline("conversational", model="facebook/blenderbot-400M-distill", device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

# Initialize Vosk model (with proper error handling)
@st.cache_resource
def load_vosk_model():
    try:
        model_path = "vosk-model-small-en-us-0.15"  # You should download this model
        if not os.path.exists(model_path):
            st.error(f"Please download the Vosk model to {model_path}")
            return None
        return Model(model_path)
    except Exception as e:
        logger.error(f"Error loading Vosk model: {str(e)}")
        return None

# Asynchronous TTS function using edge-tts
async def generate_speech(text: str) -> bytes:
    try:
        communicate = edge_tts.Communicate(text, 'en-US-ChristopherNeural')
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_path = temp_file.name
            await communicate.save(temp_path)
        
        # Convert MP3 to WAV using pydub
        audio = AudioSegment.from_mp3(temp_path)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        os.unlink(temp_path)  # Clean up temp file
        
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"TTS Error: {str(e)}")
        return None

def get_ai_response(text: str, conversation_pipeline) -> str:
    try:
        response = conversation_pipeline(text)
        return response[0]['generated_text']
    except Exception as e:
        logger.error(f"AI Response Error: {str(e)}")
        return "I apologize, but I'm having trouble generating a response right now."

class AudioProcessor:
    def __init__(self) -> None:
        self.vosk_model = load_vosk_model()
        if self.vosk_model:
            self.recognizer = KaldiRecognizer(self.vosk_model, 16000)
        else:
            st.error("Failed to load speech recognition model")
    
    def process_audio(self, frame):
        try:
            if not hasattr(self, 'recognizer'):
                return None

            # Convert audio to the correct format
            audio_data = frame.to_ndarray()
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32768).astype(np.int16)
            
            # Create WAV data in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_data.tobytes())
            
            wav_buffer.seek(0)
            
            # Process audio with Vosk
            with wave.open(wav_buffer, 'rb') as wf:
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if self.recognizer.AcceptWaveform(data):
                        result = self.recognizer.Result()
                        return result
            
            return None
            
        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}")
            return None

def display_chat_history():
    if "chat_history" in st.session_state and st.session_state["chat_history"]:
        for message in st.session_state["chat_history"]:
            with st.container():
                if message["role"] == "user":
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px;'>
                        🗣️ <b>You</b> ({message["timestamp"]})<br>{message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background-color: #e6f3ff; padding: 10px; border-radius: 5px; margin: 5px;'>
                        🤖 <b>AI</b> ({message["timestamp"]})<br>{message["content"]}
                    </div>
                    """, unsafe_allow_html=True)

def main():
    try:
        st.set_page_config(page_title="Voice Chat AI", layout="wide")
        st.title("Real-Time Voice Chat with AI")
        
        # Initialize session state
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        if "audio_processor" not in st.session_state:
            st.session_state["audio_processor"] = AudioProcessor()
        
        # Load AI model
        conversation_pipeline = load_conversation_model()
        if not conversation_pipeline:
            st.error("Failed to load AI model")
            return

        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Chat History")
            chat_container = st.container()
            with chat_container:
                display_chat_history()

        with col2:
            st.markdown("### Voice Controls")
            webrtc_ctx = webrtc_streamer(
                key="voice-chat",
                mode=WebRtcMode.SENDONLY,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ),
                media_stream_constraints={
                    "audio": True,
                    "video": False,
                },
                async_processing=True,
                video_processor_factory=None,
                audio_processor_factory=lambda: st.session_state["audio_processor"]
            )

            if st.button("Clear Chat", type="secondary"):
                st.session_state["chat_history"] = []
                st.rerun()

        # Process recognized speech
        if webrtc_ctx.state.playing:
            result = st.session_state["audio_processor"].process_audio(webrtc_ctx.audio_frame)
            if result:
                # Process the speech recognition result
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Add user message to chat
                st.session_state["chat_history"].append({
                    "role": "user",
                    "content": result["text"],
                    "timestamp": timestamp
                })
                
                # Get and add AI response
                ai_response = get_ai_response(result["text"], conversation_pipeline)
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": ai_response,
                    "timestamp": timestamp
                })
                
                # Generate speech for AI response
                audio_bytes = asyncio.run(generate_speech(ai_response))
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                
                st.rerun()

    except Exception as e:
        logger.error(f"Main application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
