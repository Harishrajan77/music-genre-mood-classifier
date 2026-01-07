#!/usr/bin/env python3
"""
Audio → YAMNet → 2048-d embedding → Genre + Mood → Curated playlists
"""

import os
import sys

# -------------------------------------------------
# Fix import path (so src.* works)
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import librosa
import joblib
import tensorflow as tf
import tensorflow_hub as hub

from src.recommend import show_recommendations

# -------------------------------------------------
# Paths
# -------------------------------------------------
MODEL_DIR = "models/embeddings_model"
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

CLASSIFIER_PATH = os.path.join(MODEL_DIR, "best_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

# -------------------------------------------------
# Genre → Mood mapping
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
# Load models
# -------------------------------------------------
def load_models():
    yamnet = hub.load(YAMNET_URL)
    classifier = tf.keras.models.load_model(CLASSIFIER_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return yamnet, classifier, scaler, label_encoder

# -------------------------------------------------
# Extract 2048-d embedding (same as training)
# -------------------------------------------------
def extract_embedding(yamnet, audio_path, sr=16000):
    wav, _ = librosa.load(audio_path, sr=sr, mono=True)
    _, embeddings, _ = yamnet(wav)

    mean_emb = np.mean(embeddings, axis=0)
    std_emb = np.std(embeddings, axis=0)

    return np.concatenate([mean_emb, std_emb])

# -------------------------------------------------
# Predict
# -------------------------------------------------
def predict(audio_path):
    yamnet, classifier, scaler, label_encoder = load_models()

    emb = extract_embedding(yamnet, audio_path)
    emb_scaled = scaler.transform([emb])

    probs = classifier.predict(emb_scaled, verbose=0)[0]
    classes = label_encoder.classes_

    top3 = probs.argsort()[-3:][::-1]

    print("\n🎯 Top Predictions:")
    for i in top3:
        print(f"  {classes[i]:<10} → {probs[i]*100:.2f}%")

    best_genre = classes[np.argmax(probs)]
    mood = GENRE_MOOD.get(best_genre, "Unknown")

    print(f"\n🔥 Final Genre: {best_genre}")
    print(f"💙 Mood: {mood}")

    # Curated playlists
    show_recommendations(best_genre)

# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    audio_path = input("🎧 Enter audio file path (.mp3/.wav): ").strip()

    if not os.path.exists(audio_path):
        print("❌ File not found.")
    else:
        predict(audio_path)
