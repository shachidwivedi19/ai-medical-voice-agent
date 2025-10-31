import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai
import tempfile
import os

# 🔐 Load Gemini API key from Streamlit secrets
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except KeyError:
    st.error("⚠️ GOOGLE_API_KEY not found in secrets. Please add it in Streamlit Cloud settings.")
    st.stop()

# 🎨 App UI setup
st.set_page_config(page_title="AI Medical Voice Agent", page_icon="🩺", layout="centered")

st.title("🩺 AI Medical Voice Agent (Gemini)")
st.caption("Upload or record your question. The AI provides safe, factual medical information — not a diagnosis or prescription.")

# 🎤 Audio input using Streamlit's native audio recorder
audio_file = st.audio_input("🎙 Record your medical question")

if audio_file is not None:
    st.success("✅ Audio received!")
    
    # Save uploaded audio to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio_file.read())
        audio_path = tmpfile.name

    # 🎤 Speech-to-text using Google Speech Recognition
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            user_text = recognizer.recognize_google(audio_data)
            st.subheader("🗣 You said:")
            st.write(user_text)
    except sr.UnknownValueError:
        st.error("⚠ Sorry, I couldn't understand your voice. Please try again.")
        os.unlink(audio_path)
        st.stop()
    except sr.RequestError:
        st.error("⚠ Speech recognition service error. Please check your connection.")
        os.unlink(audio_path)
        st.stop()
    except Exception as e:
        st.error(f"⚠ Error processing audio: {e}")
        os.unlink(audio_path)
        st.stop()

    # Clean up temp file
    os.unlink(audio_path)

    # 🧠 Generate AI response from Gemini
    with st.spinner("💬 Thinking..."):
        model_names = ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"]
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
        st.error("⚠ Could not connect to Gemini API. Please check your API key or model availability.")
        st.stop()

    # 🧾 Display AI response
    st.subheader("🤖 AI Response:")
    st.write(ai_text)

    # 🔊 Convert AI text to speech
    try:
        tts = gTTS(ai_text)
        audio_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(audio_out.name)
        
        with open(audio_out.name, 'rb') as f:
            st.audio(f.read(), format="audio/mp3")
        
        os.unlink(audio_out.name)
        st.success("🎯 Response generated successfully!")
    except Exception as e:
        st.warning(f"Speech synthesis failed: {e}. Text response is shown above.")
