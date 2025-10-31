import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import tempfile
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai
import time

# 🔐 Load Gemini API key from Streamlit secrets or fallback
genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY", None))

# 🎨 App UI setup
st.set_page_config(page_title="AI Medical Voice Agent", page_icon="🩺", layout="centered")

st.title("🩺 AI Medical Voice Agent (Gemini)")
st.caption("Speak your question. The AI provides safe, factual medical information — not a diagnosis or prescription.")

# 🎧 Recording duration setup
duration = st.slider("🎧 Recording duration (seconds):", 5, 20, 10)
fs = 44100  # Sample rate

if st.button("🎙 Record Voice"):
    st.info(f"🎙 Recording for {duration} seconds... please speak clearly.")
    
    # Record audio from the microphone
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    st.success("✅ Recording complete!")

    # Save audio to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        with wave.open(tmpfile.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(audio.tobytes())
        audio_path = tmpfile.name

    # 🎤 Speech-to-text using Google Speech Recognition
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            user_text = recognizer.recognize_google(audio_data)
            st.subheader("🗣 You said:")
            st.write(user_text)
        except sr.UnknownValueError:
            st.error("⚠ Sorry, I couldn’t understand your voice. Please try again.")
            st.stop()
        except sr.RequestError:
            st.error("⚠ Speech recognition service error. Please check your connection.")
            st.stop()

    # 🧠 Generate AI response from Gemini
    st.info("💬 Thinking...")
    model_names = ["gemini-2.0-flash", "gemini-2.0-pro", "gemini-1.5-flash", "gemini-1.5-pro"]
    ai_text = None

    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            prompt = (
                "You are a factual and safe medical information assistant. "
                "Provide helpful, general, and evidence-based health guidance. "
                "Do not diagnose or prescribe treatments.\n\n"
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

    # 🧾 Display AI response
    st.subheader("🤖 AI Response:")
    st.write(ai_text)

    # 🔊 Convert AI text to speech
    try:
        tts = gTTS(ai_text)
        audio_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(audio_out.name)
        st.audio(audio_out.name, format="audio/mp3")
        st.success("🎯 Response generated successfully!")
    except Exception:
        st.warning("Speech synthesis failed, but text response is shown above.")
