import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
import pandas as pd

RAW_DATA = "data/raw/genres_original"
EMB_DIR = "data/embeddings"
SAMPLE_RATE = 16000   # YAMNet required sample rate
DURATION = 30         # Fix audio length for consistency

os.makedirs(EMB_DIR, exist_ok=True)

print("📥 Loading YAMNet model from TensorFlow Hub...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")


def extract_embedding(audio_path):
    """Extract mean+std YAMNet embeddings (2048 size vector)."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    required_len = SAMPLE_RATE * DURATION
    if len(y) < required_len:
        y = np.pad(y, (0, required_len - len(y)))
    else:
        y = y[:required_len]

    # YAMNet forward pass
    scores, embeddings, spectrogram = yamnet_model(y)

    # Convert variable-length embedding to fixed-length vector
    emb_mean = tf.reduce_mean(embeddings, axis=0).numpy()
    emb_std = tf.math.reduce_std(embeddings, axis=0).numpy()

    return np.concatenate([emb_mean, emb_std], axis=0)  # final: 2048 dim


print("\n🔧 Extracting embeddings for dataset...\n")

rows = []
for genre in sorted(os.listdir(RAW_DATA)):
    genre_path = os.path.join(RAW_DATA, genre)
    if not os.path.isdir(genre_path):
        continue

    save_genre_dir = os.path.join(EMB_DIR, genre)
    os.makedirs(save_genre_dir, exist_ok=True)

    print(f"\n🎵 Processing genre: {genre}")

    for file in tqdm(sorted(os.listdir(genre_path))):
        if not file.endswith(".wav"):
            continue

        audio_path = os.path.join(genre_path, file)
        save_path = os.path.join(save_genre_dir, file.replace(".wav", ".npz"))

        try:
            emb = extract_embedding(audio_path)
            np.savez_compressed(save_path, emb=emb)
            rows.append({
                "filepath": audio_path,
                "genre": genre,
                "emb_path": save_path
            })
        except Exception as e:
            print(f"❌ ERROR processing {audio_path}: {e}")


# Save manifest CSV
manifest_path = "data/embeddings_manifest.csv"
pd.DataFrame(rows).to_csv(manifest_path, index=False)

print("\n✅ Embedding Extraction Complete!")
print(f"📄 Manifest saved at: {manifest_path}")
