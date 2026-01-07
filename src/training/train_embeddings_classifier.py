import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tqdm import tqdm

MANIFEST = "data/embeddings_manifest.csv"
OUT_DIR = "models/embeddings_model"
os.makedirs(OUT_DIR, exist_ok=True)

# Load manifest
df = pd.read_csv(MANIFEST)
labels = df['genre'].values
le = LabelEncoder()
y = le.fit_transform(labels)

# Load embeddings
X = []
print("📥 Loading embeddings...")
for path in tqdm(df['emb_path'].values):
    emb = np.load(path)['emb']
    X.append(emb)
X = np.vstack(X)
print("👉 Embedding shape:", X.shape)

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
joblib.dump(le, os.path.join(OUT_DIR, "label_encoder.joblib"))

NUM_CLASSES = len(le.classes_)
print("\n🎯 Classes:", le.classes_)

# Dataset split (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

def build_mlp(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_mlp(X_train.shape[1], NUM_CLASSES)
model.summary()

es = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
mc = callbacks.ModelCheckpoint(
    os.path.join(OUT_DIR, "best_model.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# Train
print("\n🚀 Training classifier...\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=32,
    callbacks=[es, mc],
    verbose=1
)

# Evaluate
print("\n📊 Evaluating best model...")
best = tf.keras.models.load_model(os.path.join(OUT_DIR, "best_model.keras"))
preds = best.predict(X_test)
y_pred = preds.argmax(axis=1)

print("\n🏆 TEST ACCURACY:", accuracy_score(y_test, y_pred))
print("\n📘 CLASSIFICATION REPORT:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("\n📉 CONFUSION MATRIX:\n", confusion_matrix(y_test, y_pred))
