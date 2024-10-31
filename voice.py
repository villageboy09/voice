import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings, WebRtcMode
import edge_tts
import asyncio
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import numpy as np
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize recognizer
recognizer = sr.Recognizer()

# Text-to-Speech function using Edge-TTS
async def tts_speak(text):
    try:
        tts = edge_tts.Communicate(text=text, voice="en-US-JennyNeural")
        await tts.save("response.mp3")
        audio_file = AudioSegment.from_mp3("response.mp3")
        audio_bytes = BytesIO()
        audio_file.export(audio_bytes, format="wav")
        return audio_bytes.getvalue()
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        logger.error(f"TTS Error: {str(e)}")
        return None

# Function to get response
def get_ai_response(text):
    # Simple response for testing
    return f"I heard you say: {text}. How can I help you further?"

# WebRTC Audio Processor
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        self.recognizer = sr.Recognizer()
    
    def recv(self, frame):
        try:
            # Convert audio frame to numpy array
            audio_data = frame.to_ndarray()
            
            # Ensure audio data is in the correct format
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32768).astype(np.int16)
            
            # Create AudioSegment
            audio_segment = AudioSegment(
                audio_data.tobytes(), 
                frame_rate=frame.sample_rate,
                sample_width=2,
                channels=1 if len(audio_data.shape) == 1 else audio_data.shape[1]
            )
            
            # Convert to format suitable for speech recognition
            wav_bytes = BytesIO()
            audio_segment.export(wav_bytes, format="wav")
            wav_bytes.seek(0)
            
            with sr.AudioFile(wav_bytes) as source:
                audio = self.recognizer.record(source)
                try:
                    text = self.recognizer.recognize_google(audio)
                    if text:
                        # Update session state if text is different
                        if "last_text" not in st.session_state or st.session_state["last_text"] != text:
                            st.session_state["last_text"] = text
                            st.session_state["user_text"] = text
                            
                            # Initialize chat history if needed
                            if "chat_history" not in st.session_state:
                                st.session_state["chat_history"] = []
                            
                            # Add user message
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            st.session_state["chat_history"].append({
                                "role": "user",
                                "content": text,
                                "timestamp": timestamp
                            })
                            
                            # Get and add AI response
                            ai_response = get_ai_response(text)
                            st.session_state["chat_history"].append({
                                "role": "assistant",
                                "content": ai_response,
                                "timestamp": timestamp
                            })
                            
                            # Queue TTS response
                            st.session_state["pending_tts"] = ai_response
                            
                            # Log successful processing
                            logger.info(f"Processed speech: {text}")
                            
                except sr.UnknownValueError:
                    pass  # Silent failure for no speech detected
                except sr.RequestError as e:
                    logger.error(f"Speech recognition error: {str(e)}")
                    
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
        # Page config
        st.set_page_config(page_title="Voice Chat AI", layout="wide")
        
        # Main title
        st.title("Real-Time Voice Chat with AI")
        st.write("Click 'START' and begin speaking to interact with the AI!")

        # Debug info in sidebar
        with st.sidebar:
            st.subheader("Debug Information")
            st.write("Session State Keys:", list(st.session_state.keys()))
        
        # Initialize session states
        if "user_text" not in st.session_state:
            st.session_state["user_text"] = ""
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        if "pending_tts" not in st.session_state:
            st.session_state["pending_tts"] = None

        # Create columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Chat container
            chat_container = st.container()
            with chat_container:
                display_chat_history()

        with col2:
            # WebRTC Streamer
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
            
            # Status indicator
            if webrtc_ctx.state.playing:
                st.success("🎤 Listening...")
            else:
                st.warning("🔇 Microphone inactive")

            # Clear chat button
            if st.button("Clear Chat History"):
                st.session_state["chat_history"] = []
                st.session_state["user_text"] = ""
                st.session_state["last_text"] = ""
                st.rerun()

        # Handle TTS response
        if st.session_state.get("pending_tts"):
            try:
                response_text = st.session_state["pending_tts"]
                audio_response = asyncio.run(tts_speak(response_text))
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
