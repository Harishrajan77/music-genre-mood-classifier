# 🎵 Music Mood Classifier

An end-to-end deep learning system that classifies music genres, detects mood, and recommends curated playlists. Built with TensorFlow, YAMNet embeddings, and Streamlit.

## System Workflow

1. **Audio Upload** → User uploads an audio file (.mp3 or .wav)
2. **Feature Extraction** → YAMNet embeddings are extracted from the audio
3. **Genre Classification** → A trained neural network predicts the music genre
4. **Mood Mapping** → The genre is mapped to a specific mood (Sad, Calm, Happy, Energetic, etc.)
5. **Playlist Generation** → Curated Spotify and YouTube playlist links are displayed based on the predicted genre

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Audio Processing** | Librosa |
| **Feature Extraction** | TensorFlow Hub (YAMNet - Google's audio model) |
| **Classification** | TensorFlow/Keras (Dense Neural Network) |
| **Preprocessing** | Scikit-learn (StandardScaler) |
| **Web Framework** | Streamlit |
| **Language** | Python 3.8+ |

## Supported Genres

Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/music-mood-classifier.git
cd music-mood-classifier

# Create virtual environment
python -m venv venv
venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Project Structure

```
music-mood-classifier/
├── app.py                                    # Main Streamlit web interface
├── requirements.txt                          # Project dependencies
├── src/
│   ├── recommend.py                         # Playlist recommendation engine
│   ├── training/
│   │   ├── extract_yamnet_embeddings.py    # YAMNet feature extraction
│   │   └── train_embeddings_classifier.py  # Model training pipeline
│   └── inference/
│       └── predict_yamnet.py                # Inference utilities
├── models/embeddings_model/
│   ├── best_model.keras                     # Trained genre classifier
│   ├── label_encoder.joblib                 # Genre label encoder
│   └── scaler.joblib                        # Feature scaler
├── data/
│   ├── embeddings_manifest.csv              # Embeddings metadata
│   ├── embeddings/                          # Pre-extracted YAMNet embeddings
│   └── raw/                                 # Original audio data
└── visualization/
    └── generate_all_reports.py              # Model evaluation reports
```

## How It Works

### Feature Extraction
- Audio files are loaded at 16kHz mono using Librosa
- YAMNet (pre-trained on Google's AudioSet) extracts embeddings
- Mean and standard deviation of embeddings are concatenated (1024-dim feature vector)

### Classification
- Features are normalized using StandardScaler
- A trained neural network classifies the audio into 10 genre categories
- Top-3 predictions with confidence scores are returned

### Genre-to-Mood Mapping
```python
Genre Mapping:
Blues → Sad/Emotional
Classical → Calm/Relaxing
Country → Warm/Sentimental
Disco → Happy/Dance
Hip-Hop → Energetic/Pump
Jazz → Smooth/Chill
Metal → Aggressive/High Energy
Pop → Feel-Good/Upbeat
Reggae → Relaxed/Positive
Rock → Energetic/Powerful
```

### Playlist Recommendations
Based on the classified genre, the system generates curated Spotify and YouTube playlist links for user exploration.

## Model Performance

The classifier is trained on 10 music genres with detailed evaluation metrics available in `yamnet_reports_fixed/classification_report.txt`

## Key Features

✅ Real-time audio analysis  
✅ Top-3 genre predictions with confidence scores  
✅ Mood classification based on genre  
✅ Curated playlist recommendations  
✅ Clean, intuitive web interface  
✅ Fast inference on CPU  

## Requirements

See `requirements.txt` for all dependencies. Key packages:
- TensorFlow 2.x
- TensorFlow Hub
- Librosa
- Streamlit
- Scikit-learn
- Joblib
- NumPy

## Notes

- Large pre-extracted embeddings and models are excluded from the repository (see `.gitignore`)
- YAMNet model is automatically downloaded from TensorFlow Hub on first run
- The classifier achieves strong performance across all 10 genres

## License

MIT License

---

**Questions or Contributions?** Feel free to open an issue or submit a pull request!
