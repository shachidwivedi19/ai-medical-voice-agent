import streamlit as st
import numpy as np
import tempfile
import wave
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# 🔐 Configure API key
genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY", None) or "AIzaSyDmhaN1ZJ1IswbVZQo62IMVCajxGZW2n1Y")

# 🎨 Streamlit UI
st.set_page_config(page_title="AI Medical Voice Agent", page_icon="🩺", layout="centered")

st.title("🩺 AI Medical Voice Agent (Gemini)")
st.caption("Ask medical-related questions safely and easily using your voice. Works on all devices.")

st.markdown("---")

# 🎙 WebRTC Audio Recorder
st.subheader("🎧 Record your voice below")

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv_audio(self, frame):
        self.frames.append(frame.to_ndarray().flatten())
        return frame

webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDRECV,
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=False,
)

if webrtc_ctx and webrtc_ctx.state.playing:
    st.info("🎙 Recording in progress... Speak now.")

if st.button("🛑 Stop & Analyze") and webrtc_ctx and webrtc_ctx.audio_receiver:
    st.info("Processing your voice...")

    # Combine frames
    audio_frames = []
    while True:
        try:
            audio_frame = webrtc_ctx.audio_receiver.get_frame(timeout=1)
            audio_frames.append(audio_frame.to_ndarray().flatten())
        except:
            break

    if len(audio_frames) == 0:
        st.warning("No audio recorded.")
        st.stop()

    audio = np.concatenate(audio_frames, axis=0)
    fs = 48000  # WebRTC default sample rate

    # Save to WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        with wave.open(tmpfile.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(audio.astype(np.int16).tobytes())
        audio_path = tmpfile.name

    # 🎧 Speech recognition
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            user_text = recognizer.recognize_google(audio_data)
            st.subheader("🗣 You said:")
            st.write(user_text)
        except sr.UnknownValueError:
            st.error("Sorry, I couldn't understand your voice. Please try again.")
            st.stop()

    # 🧠 Gemini response
    model_names = ["gemini-2.0-flash", "gemini-2.0-pro", "gemini-1.5-flash"]
    ai_text = None

    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            prompt = (
                "You are a safe, factual medical assistant. "
                "Give general information only (no diagnosis or prescriptions).\n\n"
                f"User said: {user_text}"
            )
            response = model.generate_content(prompt)
            ai_text = response.text
            break
        except Exception:
            continue

    if ai_text:
        st.subheader("🤖 AI Response:")
        st.write(ai_text)

        # 🔊 Text-to-speech output
        tts = gTTS(ai_text)
        audio_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(audio_out.name)
        st.audio(audio_out.name, format="audio/mp3")

        st.success("🎯 Response generated successfully!")
    else:
        st.error("⚠ Unable to get a response from Gemini. Try again later.")

