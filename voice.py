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

def main():
    try:
        # Streamlit UI
        st.title("Real-Time Voice Chat with AI")
        st.write("Start speaking to interact with the AI!")
        
        # Add debug information
        st.write("Debug Information:")
        st.write("- Session State:", list(st.session_state.keys()))
        
        # Initialize session states
        if "user_text" not in st.session_state:
            st.session_state["user_text"] = ""
            logger.info("Initialized user_text in session state")
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
            logger.info("Initialized chat_history in session state")
        if "pending_tts" not in st.session_state:
            st.session_state["pending_tts"] = None
            logger.info("Initialized pending_tts in session state")

        # Create a container for the chat interface
        chat_container = st.container()

        # WebRTC Streamer configuration
        try:
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
            logger.info("WebRTC streamer initialized successfully")
        except Exception as e:
            st.error(f"WebRTC Error: {str(e)}")
            logger.error(f"WebRTC initialization failed: {str(e)}")

        # Display chat history
        with chat_container:
            display_chat_history()
            
        # Add status indicators
        st.sidebar.write("Status:")
        st.sidebar.write("- WebRTC Connected:", bool(webrtc_ctx.state.playing))
        
        # Handle TTS response
        if st.session_state.get("pending_tts"):
            try:
                response_text = st.session_state["pending_tts"]
                st.info("Generating audio response...")
                audio_response = asyncio.run(tts_speak(response_text))
                if audio_response:
                    st.audio(audio_response, format="audio/wav", start_time=0)
                st.session_state["pending_tts"] = None
            except Exception as e:
                st.error(f"TTS Error: {str(e)}")
                logger.error(f"TTS processing failed: {str(e)}")

        # Add a clear chat button
        if st.button("Clear Chat History"):
            st.session_state["chat_history"] = []
            st.session_state["user_text"] = ""
            st.session_state["last_text"] = ""
            logger.info("Chat history cleared")
            st.rerun()

    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        logger.error(f"Main application error: {str(e)}")

if __name__ == "__main__":
    main()
