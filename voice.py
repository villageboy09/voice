import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
import edge_tts
import asyncio
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO

# Initialize recognizer
recognizer = sr.Recognizer()

# Text-to-Speech function using Edge-TTS
async def tts_speak(text):
    tts = edge_tts.Communicate(text, voice="en-US-JennyNeural")  # Change language if needed
    await tts.save("response.mp3")
    audio_file = AudioSegment.from_mp3("response.mp3")
    audio_bytes = BytesIO()
    audio_file.export(audio_bytes, format="wav")
    return audio_bytes.getvalue()

# WebRTC Audio Processor for Voice Recognition
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = sr.Recognizer()
    
    def recv(self, frame):
        audio = frame.to_ndarray().astype("int16")
        audio_segment = AudioSegment(
            audio.tobytes(), 
            frame_rate=frame.sample_rate, 
            sample_width=frame.sample_width,
            channels=frame.channels
        )
        with sr.AudioFile(BytesIO(audio_segment.raw_data)) as source:
            audio_data = self.recognizer.record(source)
            try:
                # Recognize speech
                text = self.recognizer.recognize_google(audio_data)
                st.session_state["user_text"] = text
                return text
            except sr.UnknownValueError:
                st.error("Sorry, could not understand the audio.")
                return ""
            except sr.RequestError:
                st.error("Speech recognition service error.")
                return ""

# Streamlit UI for Voice Interaction
st.title("Real-Time Voice Interaction with NLP")
st.write("Start speaking to interact with the AI!")

if "user_text" not in st.session_state:
    st.session_state["user_text"] = ""

# WebRTC Streamer for capturing audio
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings

# WebRTC Streamer for capturing audio
webrtc_ctx = webrtc_streamer(
    key="voice",
    mode="sendonly",  # "sendonly" passed directly as a string
    audio_processor_factory=AudioProcessor,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
)


# Process the response
if st.session_state["user_text"]:
    user_text = st.session_state["user_text"]
    st.write("You said:", user_text)
    
    response_text = f"AI response: {user_text}"  # Placeholder for real NLP model response
    st.write("Generating response...")

    # Generate and play TTS response
    audio_response = asyncio.run(tts_speak(response_text))
    st.audio(audio_response, format="audio/wav")
