import streamlit as st
import tempfile
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from streamlit_audio_recorder import audio_recorder
import google.generativeai as genai

# 🔐 Gemini API key setup
genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY", None) or "YOUR_GEMINI_API_KEY_HERE")

# 🎨 Streamlit UI setup
st.set_page_config(page_title="AI Medical Voice Agent", page_icon="🩺", layout="centered")
st.title("🩺 AI Medical Voice Agent (Gemini)")
st.caption("Speak for 10 seconds — AI will listen, transcribe, and respond with safe, factual medical information (not diagnosis).")

# 🎙 Voice recording
st.subheader("🎤 Click below to record your voice:")
audio_bytes = audio_recorder(pause_threshold=10.0, sample_rate=44100)

if audio_bytes:
    st.success("✅ Recording complete!")

    # Save to a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        audio_path = temp_audio.name

    # 🧠 Convert speech to text
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            user_text = recognizer.recognize_google(audio_data)
            st.subheader("🗣 You said:")
            st.write(user_text)
        except sr.UnknownValueError:
            st.error("⚠️ Sorry, I couldn’t understand your voice. Please try again.")
            st.stop()

    # 🤖 Gemini AI Response
    st.subheader("🤖 AI Medical Response:")
    model_names = ["gemini-2.0-flash", "gemini-2.0-pro", "gemini-1.5-pro", "gemini-1.5-flash"]
    ai_text = None

    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            prompt = (
                "You are a safe and factual AI medical information assistant. "
                "Provide general, educational medical guidance only. "
                "Do not diagnose or prescribe.\n\n"
                f"Patient asked: {user_text}"
            )
            response = model.generate_content(prompt)
            ai_text = response.text
            break
        except Exception:
            continue

    if not ai_text:
        st.error("⚠️ Could not connect to Gemini API. Check your API key or model name.")
        st.stop()

    st.write(ai_text)

    # 🔊 Convert AI text to speech
    tts = gTTS(ai_text)
    tts_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tts_path.name)
    st.audio(tts_path.name, format="audio/mp3")

    st.success("🎯 Response generated successfully!")

else:
    st.info("🎧 Click the mic icon above and speak clearly for about 10 seconds.")
