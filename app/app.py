import streamlit as st
import librosa
import numpy as np
from transformers import pipeline
import language_tool_python
import tempfile
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import librosa.display
import re

# 1. Advanced Caching
@st.cache_resource
def load_resources():
    return {
        'asr': pipeline("automatic-speech-recognition", model="openai/whisper-small"),
        'grammar': language_tool_python.LanguageTool('en-US', remote_server='https://api.languagetool.org'),
        'model': tf.keras.models.load_model('models/hybrid_model.h5'),
        'scaler_audio': joblib.load('models/scaler_audio.pkl'),
        'scaler_text': joblib.load('models/scaler_text.pkl')
    }

# 2. Feature Visualization
def plot_audio_analysis(y, sr):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))

    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set_title('Waveform')

    # Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
    fig.colorbar(img, ax=ax[1], format="%+2.f dB")
    ax[1].set_title('Spectrogram')

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax[2])
    fig.colorbar(img, ax=ax[2])
    ax[2].set_title('MFCC')

    return fig

# 3. Grammar Analysis
def analyze_grammar(text, resources):
    sentences = re.split(r'(?<=[.!?]) +', text)
    total_matches = 0
    unique_errors = set()

    for sentence in sentences:
        try:
            if len(sentence.split()) < 3:
                continue
            matches = resources['grammar'].check(sentence)
            total_matches += len(matches)
            unique_errors.update(m.ruleId for m in matches)
        except:
            continue

    return {
        'error_count': total_matches,
        'error_types': unique_errors
    }

# 4. Extract Audio Features (40 MFCC + 12 Chroma + 4 Contrast = 56 features)
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    feature_vector = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(contrast, axis=1)
    ])
    return feature_vector.reshape(1, -1)  # Shape: (1, 56)

# 5. Extract Text Features (2 features: error count and unique error types)
def extract_text_features(grammar_report):
    return np.array([[grammar_report['error_count'], len(grammar_report['error_types'])]])

# 6. Main Application
def main():
    st.title("🎙️ Grammar Scoring Engine")

    uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())

            y, sr = librosa.load(tmp_file.name, sr=16000)
            st.subheader("Audio Analysis")
            st.pyplot(plot_audio_analysis(y, sr))

            resources = load_resources()

            # Transcription
            transcription = resources['asr'](tmp_file.name)["text"]
            st.subheader("Transcription")
            st.code(transcription)

            # Grammar
            grammar_report = analyze_grammar(transcription, resources)
            st.subheader("Grammar Report")
            col1, col2 = st.columns(2)
            col1.metric("Total Errors", grammar_report['error_count'])
            col2.metric("Unique Error Types", len(grammar_report['error_types']))

            # Feature Extraction
            audio_features = extract_audio_features(tmp_file.name)
            text_features = extract_text_features(grammar_report)

            # Scaling
            audio_scaled = resources['scaler_audio'].transform(audio_features)
            text_scaled = resources['scaler_text'].transform(text_features)

            # Prediction
            prediction = resources['model'].predict([audio_scaled, text_scaled])
            score = max(0.0, min(5.0, prediction[0][0]))  # Simple clipping

            st.subheader("Predicted Grammar Score")
            st.metric("Score", f"{score:.2f} / 5.0")
            clipped_score = np.clip(score, 0.0, 5.0)  # Ensure it's within valid range
            st.progress(clipped_score / 5.0)


if __name__ == "__main__":
    main()
