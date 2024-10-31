import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings, WebRtcMode
import edge_tts
import asyncio
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import numpy as np

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
        return None

# WebRTC Audio Processor for Voice Recognition
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
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
                sample_width=2,  # 16-bit audio
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
                        st.session_state["user_text"] = text
                        return frame  # Return the original frame
                except sr.UnknownValueError:
                    pass  # Silent failure for no speech detected
                except sr.RequestError as e:
                    st.error(f"Speech recognition error: {str(e)}")
                
            return frame  # Return the original frame
            
        except Exception as e:
            st.error(f"Audio processing error: {str(e)}")
            return frame

def main():
    # Streamlit UI
    st.title("Real-Time Voice Interaction with NLP")
    st.write("Start speaking to interact with the AI!")

    # Initialize session state
    if "user_text" not in st.session_state:
        st.session_state["user_text"] = ""

    # WebRTC Streamer configuration
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

    # Display user's speech and generate response
    if st.session_state["user_text"]:
        user_text = st.session_state["user_text"]
        st.write("You said:", user_text)
        
        # Generate AI response (placeholder - replace with your NLP model)
        response_text = f"I heard you say: {user_text}"
        st.write("AI Response:", response_text)
        
        # Generate and play TTS response
        if st.button("Play Response"):
            st.write("Generating audio response...")
            audio_response = asyncio.run(tts_speak(response_text))
            if audio_response:
                st.audio(audio_response, format="audio/wav")

if __name__ == "__main__":
    main()
