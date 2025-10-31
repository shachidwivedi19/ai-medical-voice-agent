import streamlit as st
from st_audiorec import st_audiorec
import tempfile
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

st.set_page_config(page_title="🩺 AI Medical Voice Agent", page_icon="🎙", layout="centered")
st.title("🩺 AI Medical Voice Agent (Gemini)")
st.caption("Speak for a few seconds — AI will listen, transcribe, and respond with safe, factual medical info.")

st.info("🎤 Click below to record your voice (works on all devices).")

# 🎧 Record audio
audio_bytes = st_audiorec()

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    st.success("✅ Recording complete! Processing...")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_bytes)
        audio_path = tmpfile.name

    # 🎙 Speech Recognition
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            user_text = recognizer.recognize_google(audio_data)
            st.subheader("🗣 You said:")
            st.write(user_text)
        except sr.UnknownValueError:
            st.error("❌ Could not understand your speech.")
            st.stop()

    # 🤖 Gemini AI
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        f"You are a safe and factual medical assistant. Provide general health info for: {user_text}"
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
