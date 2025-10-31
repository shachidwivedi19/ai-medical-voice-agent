import streamlit as st
import numpy as np
import wave
import tempfile
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai

# 🔐 Configure Gemini API Key
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# 🎨 App UI
st.set_page_config(page_title="AI Medical Voice Agent", page_icon="🩺", layout="centered")
st.title("🩺 AI Medical Voice Agent (Gemini)")
st.caption("Speak your question or upload audio — the AI will listen and provide safe, factual medical info.")

# 🎙 Upload recorded audio
uploaded_audio = st.file_uploader("🎧 Upload your voice (WAV or MP3)", type=["wav", "mp3"])

if uploaded_audio is not None:
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(uploaded_audio.read())
        audio_path = tmpfile.name

    # 🧠 Speech-to-text
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            user_text = recognizer.recognize_google(audio_data)
            st.subheader("🗣 You said:")
            st.write(user_text)
        except sr.UnknownValueError:
            st.error("Sorry, I couldn’t understand your voice. Please try again.")
            st.stop()

    # 🤖 Generate AI Response
    st.info("🧠 Thinking...")
    model_name_list = ["gemini-2.0-flash", "gemini-2.0-pro", "gemini-1.5-flash", "gemini-1.5-pro"]

    ai_text = None
    for model_name in model_name_list:
        try:
            model = genai.GenerativeModel(model_name)
            prompt = (
                "You are a factual and safe medical information assistant. "
                "Provide general, evidence-based health guidance. Do not diagnose or prescribe.\n\n"
                f"Patient said: {user_text}"
            )
            response = model.generate_content(prompt)
            ai_text = response.text
            break
        except Exception:
            continue

    if not ai_text:
        st.error("⚠ Could not connect to Gemini API. Please check your API key or model name.")
        st.stop()

    st.subheader("🤖 AI Response:")
    st.write(ai_text)

    # 🔊 Text-to-Speech Response
    tts = gTTS(ai_text)
    audio_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(audio_out.name)
    st.audio(audio_out.name, format="audio/mp3")

    st.success("✅ Response generated successfully!")

else:
    st.info("Please upload a short voice recording to begin (max 10 seconds).")
