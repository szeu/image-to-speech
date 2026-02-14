# app.py
import streamlit as st
from transformers import pipeline
from PIL import Image
import io

# Page config
st.set_page_config(page_title="Vision Assistant", page_icon="🦯", layout="centered")

st.title("🗣️ Vision Assistant for the Blind")
st.markdown("**Take a photo → AI describes the objects & scene → Speaks it out loud**")

# Cache the models (loads only once)
@st.cache_resource
def load_caption_model():
    return pipeline("image-to-text", 
                    model="Salesforce/blip-image-captioning-base",
                    device="cpu")   # Change to "cuda" if you have GPU

@st.cache_resource
def load_tts_model():
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")

captioner = load_caption_model()
tts_pipe = load_tts_model()

# Webcam Capture
photo = st.camera_input("**Take a Photo**", label_visibility="visible")

if photo is not None:
    image = Image.open(photo)
    st.image(image, caption="📸 Captured Photo", use_column_width=True)

    with st.spinner("Analyzing image... (may take 6-15 seconds)"):
        result = captioner(image, max_new_tokens=80)
        description = result[0]['generated_text'].strip().capitalize() + "."

    st.subheader("📝 AI Description:")
    st.success(description)

    # ==================== Text-to-Speech ====================
    with st.spinner("Converting to speech..."):
        speech = tts_pipe(description)
        audio_bytes = io.BytesIO(speech["audio"][0].tobytes())  # Convert to bytes
        audio_bytes.seek(0)

    st.audio(audio_bytes, format="audio/wav", sample_rate=speech["sampling_rate"])
    st.info("🔊 Audio is playing automatically")

    if st.button("🔊 Speak Again"):
        st.audio(audio_bytes, format="audio/wav", sample_rate=speech["sampling_rate"])

else:
    st.info("👆 Click the camera button above to start")

st.caption("Model: BLIP Image Captioning & MMS TTS (Hugging Face)")
