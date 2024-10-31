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
import requests
import zipfile
from typing import Optional, Tuple
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
SESSION_TIMEOUT = 3600  # 1 hour
MESSAGE_LIMIT = 100
RETRY_LIMIT = 3

def download_vosk_model() -> bool:
    """
    Download and extract the Vosk model if it doesn't exist.
    Returns True if successful, False otherwise.
    """
    try:
        if not os.path.exists(VOSK_MODEL_PATH):
            st.info("Downloading Vosk model... This may take a few minutes.")
            
            # Download the model
            response = requests.get(VOSK_MODEL_URL, stream=True)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_path = temp_file.name
            
            # Extract the model
            with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                zip_ref.extractall('.')
            
            # Clean up
            os.unlink(temp_path)
            return True
            
        return True
    except Exception as e:
        logger.error(f"Error downloading Vosk model: {str(e)}")
        st.error("Failed to download speech recognition model. Please check your internet connection.")
        return False

@st.cache_resource
def load_conversation_model():
    """Initialize AI response pipeline with proper error handling."""
    try:
        return pipeline(
            "conversational",
            model="facebook/blenderbot-400M-distill",
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        logger.error(f"Error loading conversation model: {str(e)}")
        st.error("Failed to load AI model. Please check your internet connection.")
        return None

@st.cache_resource
def load_vosk_model() -> Optional[Model]:
    """Initialize Vosk model with download capability."""
    try:
        if not download_vosk_model():
            return None
        return Model(VOSK_MODEL_PATH)
    except Exception as e:
        logger.error(f"Error loading Vosk model: {str(e)}")
        st.error("Failed to load speech recognition model. Please check the logs.")
        return None

def check_ffmpeg() -> bool:
    """Verify ffmpeg installation."""
    try:
        ffmpeg.probe('null')
        return True
    except ffmpeg.Error:
        st.error("ffmpeg is not installed. Please install it to enable audio processing.")
        return False
    except Exception as e:
        logger.error(f"Error checking ffmpeg: {str(e)}")
        return False

async def generate_speech(text: str) -> Optional[bytes]:
    """Generate speech from text with retry logic."""
    for attempt in range(RETRY_LIMIT):
        try:
            communicate = edge_tts.Communicate(text, 'en-US-ChristopherNeural')
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
                await communicate.save(temp_path)
            
            audio = AudioSegment.from_mp3(temp_path)
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            os.unlink(temp_path)
            
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"TTS Error (attempt {attempt + 1}/{RETRY_LIMIT}): {str(e)}")
            if attempt == RETRY_LIMIT - 1:
                st.warning("Unable to generate speech. Falling back to text only.")
                return None
            await asyncio.sleep(1)  # Wait before retry

def get_ai_response(text: str, conversation_pipeline) -> Tuple[str, bool]:
    """Get AI response with error handling."""
    try:
        response = conversation_pipeline(text)
        return response[0]['generated_text'], True
    except Exception as e:
        logger.error(f"AI Response Error: {str(e)}")
        return "I apologize, but I'm having trouble generating a response right now.", False

class AudioProcessor:
    def __init__(self) -> None:
        self.vosk_model = load_vosk_model()
        if self.vosk_model:
            self.recognizer = KaldiRecognizer(self.vosk_model, 16000)
        self.last_process_time = time.time()
        self.min_process_interval = 1.0  # Minimum time between processing in seconds
    
    def audio_meter(self, audio_data: np.ndarray) -> float:
        """Calculate audio volume level."""
        return float(np.abs(audio_data).mean())
    
    def process_audio(self, frame) -> Optional[dict]:
        """Process audio frame with rate limiting and error handling."""
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_process_time < self.min_process_interval:
                return None
            
            if not hasattr(self, 'recognizer'):
                return None

            # Convert audio to the correct format
            audio_data = frame.to_ndarray()
            
            # Update volume meter
            volume = self.audio_meter(audio_data)
            st.session_state['current_volume'] = volume
            
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
                        self.last_process_time = current_time
                        return eval(self.recognizer.Result())
            
            return None
            
        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}")
            return None

def init_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "audio_processor" not in st.session_state:
        st.session_state["audio_processor"] = AudioProcessor()
    if "last_activity" not in st.session_state:
        st.session_state["last_activity"] = datetime.now()
    if "message_count" not in st.session_state:
        st.session_state["message_count"] = 0
    if "current_volume" not in st.session_state:
        st.session_state["current_volume"] = 0.0

def display_chat_history():
    """Display chat history with custom styling."""
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

def check_session_timeout() -> bool:
    """Check if session has timed out."""
    if (datetime.now() - st.session_state["last_activity"]).seconds > SESSION_TIMEOUT:
        st.session_state.clear()
        st.warning("Session expired. Please refresh the page.")
        return True
    return False

def check_message_limit() -> bool:
    """Check if message limit has been reached."""
    if st.session_state["message_count"] > MESSAGE_LIMIT:
        st.error("Message limit reached. Please start a new session.")
        return True
    return False

def main():
    try:
        st.set_page_config(page_title="Voice Chat AI", layout="wide")
        st.title("Real-Time Voice Chat with AI")
        
        # Initialize session state
        init_session_state()
        
        # Check session timeout
        if check_session_timeout():
            return
        
        # Check message limit
        if check_message_limit():
            return
        
        # Check ffmpeg installation
        if not check_ffmpeg():
            return
        
        # Load AI model
        conversation_pipeline = load_conversation_model()
        if not conversation_pipeline:
            return

        # Layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Chat History")
            chat_container = st.container()
            with chat_container:
                display_chat_history()

        with col2:
            st.markdown("### Voice Controls")
            
            # Display volume meter
            st.metric("Microphone Level", f"{st.session_state['current_volume']:.2f}")
            
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
                st.session_state["message_count"] = 0
                st.rerun()

        # Process recognized speech
        if webrtc_ctx.state.playing:
            result = st.session_state["audio_processor"].process_audio(webrtc_ctx.audio_frame)
            if result and "text" in result and result["text"].strip():
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Add user message to chat
                st.session_state["chat_history"].append({
                    "role": "user",
                    "content": result["text"],
                    "timestamp": timestamp
                })
                
                # Get and add AI response
                ai_response, success = get_ai_response(result["text"], conversation_pipeline)
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": ai_response,
                    "timestamp": timestamp
                })
                
                if success:
                    # Generate speech for AI response
                    audio_bytes = asyncio.run(generate_speech(ai_response))
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav")
                
                st.session_state["message_count"] += 1
                st.session_state["last_activity"] = datetime.now()
                
                st.rerun()

    except Exception as e:
        logger.error(f"Main application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
