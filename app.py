import os
import sys
import tempfile
import numpy as np
import librosa
import joblib
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st

# -------------------------------------------------
# Fix import path
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.recommend import show_recommendations

# -------------------------------------------------
# App configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Music Mood Classifier",
    page_icon="🎵",
    layout="centered"
)

# -------------------------------------------------
# Paths
# -------------------------------------------------
MODEL_DIR = "models/embeddings_model"
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

CLASSIFIER_PATH = os.path.join(MODEL_DIR, "best_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

# -------------------------------------------------
# Genre → Mood
# -------------------------------------------------
GENRE_MOOD = {
    "blues": "Sad / Emotional",
    "classical": "Calm / Relaxing",
    "country": "Warm / Sentimental",
    "disco": "Happy / Dance",
    "hiphop": "Energetic / Pump",
    "jazz": "Smooth / Chill",
    "metal": "Aggressive / High Energy",
    "pop": "Feel-Good / Upbeat",
    "reggae": "Relaxed / Positive",
    "rock": "Energetic / Powerful"
}

# -------------------------------------------------
# Load models (cached)
# -------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    yamnet = hub.load(YAMNET_URL)
    classifier = tf.keras.models.load_model(CLASSIFIER_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return yamnet, classifier, scaler, label_encoder

# -------------------------------------------------
# Feature extraction
# -------------------------------------------------
def extract_embedding(yamnet, audio_path, sr=16000):
    wav, _ = librosa.load(audio_path, sr=sr, mono=True)
    _, embeddings, _ = yamnet(wav)
    mean_emb = np.mean(embeddings, axis=0)
    std_emb = np.std(embeddings, axis=0)
    return np.concatenate([mean_emb, std_emb])

# -------------------------------------------------
# Prediction
# -------------------------------------------------
def predict(audio_path):
    yamnet, classifier, scaler, label_encoder = load_models()

    emb = extract_embedding(yamnet, audio_path)
    emb_scaled = scaler.transform([emb])

    probs = classifier.predict(emb_scaled, verbose=0)[0]
    classes = label_encoder.classes_

    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [(classes[i], probs[i]) for i in top3_idx]

    best_genre = classes[np.argmax(probs)]
    mood = GENRE_MOOD.get(best_genre, "Unknown")

    return best_genre, mood, top3

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🎵 Music Genre & Mood Classifier")
st.caption("Upload an audio file to predict its genre, mood, and get curated playlists.")

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload audio file (.mp3 or .wav)",
    type=["mp3", "wav"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(audio_path)

    col1, col2 = st.columns([1, 3])
    with col1:
        predict_btn = st.button("🔍 Predict", use_container_width=True)

    if predict_btn:
        with st.spinner("Analyzing audio…"):
            genre, mood, top3 = predict(audio_path)

        st.success("Prediction completed")

        # ---------------- Results ----------------
        st.markdown("### 🎯 Top Predictions")
        for g, p in top3:
            st.write(f"**{g}** — {p*100:.2f}%")

        st.markdown("###  Final Result")
        st.write(f"**Genre:** {genre}")
        st.write(f"**Mood:** {mood}")

        st.markdown("### 🎵 Curated Playlists")
        show_recommendations(genre)

    os.remove(audio_path)

else:
    st.info("👆 Upload an audio file to get started")
