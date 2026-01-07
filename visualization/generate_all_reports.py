import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ---------- CONFIG (edited to match your screenshot paths) ----------
MODEL_DIR = "models/embeddings_model"
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
EMB_MANIFEST = "data/embeddings_manifest.csv"   # keeps same as before
SAVE_DIR = "yamnet_reports_fixed"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- SANITY CHECKS ----------
missing = []
for p in [KERAS_MODEL_PATH, LABEL_ENCODER_PATH, SCALER_PATH, EMB_MANIFEST]:
    if not os.path.exists(p):
        missing.append(p)
if missing:
    print("❌ Missing required files. Please check these paths:")
    for m in missing:
        print("   ", m)
    raise SystemExit(1)

print("✅ Found model & artifacts. Loading now...")

# ---------- LOAD ARTIFACTS ----------
import tensorflow as tf
model = tf.keras.models.load_model(KERAS_MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)  # sklearn LabelEncoder instance
scaler = joblib.load(SCALER_PATH)               # sklearn Scaler

# label names
if hasattr(label_encoder, "classes_"):
    labels = list(label_encoder.classes_)
else:
    # fallback if joblib saved a plain list
    labels = list(label_encoder)

print("Labels:", labels)

# ---------- LOAD EMBEDDINGS MANIFEST ----------
df = pd.read_csv(EMB_MANIFEST)
print(f"Loaded manifest with {len(df)} rows")

X = []
y_true = []

# assume the stored key in .npz is "emb" (we debugged earlier)
for i, row in df.iterrows():
    emb_path = row["emb_path"]
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embedding file not found: {emb_path}")
    arr = np.load(emb_path)
    # find key
    keys = arr.files
    if "emb" in keys:
        emb = arr["emb"]
    elif "embedding" in keys:
        emb = arr["embedding"]
    elif "arr_0" in keys:
        emb = arr["arr_0"]
    else:
        # pick first available
        emb = arr[keys[0]]
    X.append(emb)
    y_true.append(row["genre"])

X = np.array(X)
y_true = np.array(y_true)
print("X shape:", X.shape)

# ---------- SCALE & PREDICT ----------
X_scaled = scaler.transform(X)

# model might be a Keras model (softmax output)
pred_probs = model.predict(X_scaled, verbose=0)
if pred_probs.ndim == 1:
    # binary probabilities -> convert to 2-class prediction
    y_pred_idx = (pred_probs > 0.5).astype(int)
else:
    y_pred_idx = np.argmax(pred_probs, axis=1)

# If label_encoder maps classes -> we need to invert
if hasattr(label_encoder, "inverse_transform"):
    try:
        y_pred = label_encoder.inverse_transform(y_pred_idx)
    except Exception:
        # sometimes label_encoder expects 2D; fallback map manually
        y_pred = np.array([labels[i] for i in y_pred_idx])
else:
    y_pred = np.array([labels[i] for i in y_pred_idx])

# ---------- METRICS & REPORT ----------
print("Computing metrics...")
report_text = classification_report(y_true, y_pred, target_names=labels)
with open(os.path.join(SAVE_DIR, "classification_report.txt"), "w") as f:
    f.write(report_text)
print("\nClassification report saved.")

# confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

plt.figure(figsize=(10, 8))
disp.plot(cmap="Blues", xticks_rotation=45, values_format='d')
plt.title("Confusion Matrix - YAMNet Embedding Model")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=300)
plt.close()
print("Confusion matrix saved.")

# parse classification_report into dict to plot per-class metrics
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

# precision plot
plt.figure(figsize=(10, 5))
plt.bar(labels, precision, color="skyblue")
plt.ylim(0, 1.0)
plt.xticks(rotation=45)
plt.title("Precision per Class")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "precision.png"), dpi=300)
plt.close()

# recall plot
plt.figure(figsize=(10, 5))
plt.bar(labels, recall, color="orange")
plt.ylim(0, 1.0)
plt.xticks(rotation=45)
plt.title("Recall per Class")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "recall.png"), dpi=300)
plt.close()

# f1-score plot
plt.figure(figsize=(10, 5))
plt.bar(labels, f1, color="green")
plt.ylim(0, 1.0)
plt.xticks(rotation=45)
plt.title("F1-score per Class")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "f1_score.png"), dpi=300)
plt.close()

print("Per-class metric plots saved.")

# feature importance (best-effort)
# Keras models do not expose feature importance; this will try to get permutation-style importances if possible
fi_path = os.path.join(SAVE_DIR, "feature_importance.png")
try:
    # If model is boosting-like saved as Keras (rare), try attribute
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        plt.figure(figsize=(12, 4))
        plt.plot(importances)
        plt.title("Feature importance")
        plt.tight_layout()
        plt.savefig(fi_path, dpi=300)
        plt.close()
        print("Feature importance saved (from model.feature_importances_).")
    else:
        # fallback: skip, but create an empty placeholder figure
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "Feature importance not available for Keras model", ha="center", va="center")
        plt.axis("off")
        plt.savefig(fi_path, dpi=300)
        plt.close()
        print("Feature importance placeholder saved.")
except Exception as e:
    print("Could not compute feature importance:", e)

print("\n✅ All reports saved to:", SAVE_DIR)
