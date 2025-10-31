import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import numpy as np
import speech_recognition as sr
from gtts import gTTS
import tempfile
import google.generativeai as genai
import wave

# 🔐 Configure Gemini API Key
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

st.set_page_config(page_title="🩺 AI Medical Voice Agent", page_icon="🎙", layout="centered")
st.title("🩺 AI Medical Voice Agent (Gemini)")
st.caption("Speak for 10 seconds — AI will listen, transcribe, and respond with safe, factual medical info.")

st.info("🎤 Click below to record your voice:")

# 🎧 Audio recording
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv_audio(self, frame):
        self.frames.append(frame.to_ndarray().flatten())
        return frame

webrtc_ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

if st.button("🛑 Stop and Process Audio"):
    if not webrtc_ctx or not webrtc_ctx.audio_receiver:
        st.warning("⚠️ No audio recorded.")
    else:
        audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=3)
        if not audio_frames:
            st.warning("⚠️ No audio captured.")
        else:
            audio_data = np.concatenate([f.to_ndarray().flatten() for f in audio_frames])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                with wave.open(tmpfile.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(44100)
                    wf.writeframes(audio_data.tobytes())
                audio_path = tmpfile.name

            st.success("✅ Recording saved! Transcribing...")

            # 🎙 Speech recognition
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                try:
                    user_text = recognizer.recognize_google(audio_data)
                    st.subheader("🗣 You said:")
                    st.write(user_text)
                except sr.UnknownValueError:
                    st.error("Sorry, I couldn’t understand your voice.")
                    st.stop()

            # 🤖 Gemini AI
            st.info("🤖 Thinking...")
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(
                f"You are a safe medical assistant. "
                f"Provide general information based on this: {user_text}"
            )
            ai_text = response.text

            st.subheader("🤖 AI Response:")
            st.write(ai_text)

            # 🔊 Text-to-speech
            tts = gTTS(ai_text)
            audio_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(audio_out.name)
            st.audio(audio_out.name, format="audio/mp3")

            st.success("🎯 Response ready!")
