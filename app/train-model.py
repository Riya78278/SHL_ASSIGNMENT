import pandas as pd
import numpy as np
import librosa
from transformers import pipeline
import language_tool_python
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import torch
from tqdm import tqdm
import re

# Load CSV Data
train_df = pd.read_csv('data/train.csv')

# Initialize ASR and Grammar Tools
if torch.cuda.is_available():
    asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0)
    print("✅ Using GPU for processing!")
else:
    asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    print("⚠️ Using CPU for processing (GPU not available)")

# Use public LanguageTool API or local one if set up
tool = language_tool_python.LanguageTool('en-US', remote_server='https://api.languagetool.org')

# --- Audio Feature Extraction ---
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    features = {
        'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40),
        'chroma': librosa.feature.chroma_stft(y=y, sr=sr),
        'contrast': librosa.feature.spectral_contrast(y=y, sr=sr)
    }
    # Instead of returning a dictionary, return a flattened numpy array of feature means
    return np.concatenate([np.mean(v, axis=1) for v in features.values()])

# --- Audio Transcription ---
def transcribe_audio(file_path):
    try:
        return asr_pipe(file_path)["text"]
    except Exception as e:
        print(f"❌ Error transcribing {file_path}: {e}")
        return ""

# --- Grammar Error Analysis ---
def analyze_grammar(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    total_matches = 0
    unique_errors = set()

    for sentence in sentences:
        try:
            if len(sentence.split()) < 3:
                continue  # skip trivial ones
            matches = tool.check(sentence)
            total_matches += len(matches)
            unique_errors.update(m.ruleId for m in matches)
        except language_tool_python.LanguageToolError as e:
            print(f"⚠️ Skipped problematic sentence: {sentence[:60]}... due to: {e}")
            continue

    return {
        'error_count': total_matches,
        'error_types': unique_errors
    }

# --- Feature Extraction ---
def extract_features_from_data(df, audio_dir):
    audio_features = []
    text_features = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_file = f"{audio_dir}/{row['filename']}"

        transcription = transcribe_audio(audio_file)
        if not transcription or len(transcription.split()) > 300:
            print(f"⚠️ Skipping transcription (too long or empty): {row['filename']}")
            continue

        grammar_report = analyze_grammar(transcription)

        audio_feat = extract_audio_features(audio_file)
        audio_features.append(audio_feat)

        text_features.append([
            grammar_report['error_count'],  # Numeric feature
            len(grammar_report['error_types'])  # Numeric feature
        ])

    return np.array(audio_features), np.array(text_features)

# Preprocess the features
audio_train_features, text_train_features = extract_features_from_data(train_df, 'data/audios_train')


# Split data into X and y
X_train_audio = audio_train_features
X_train_text = text_train_features

y_train = train_df['label'].values  # Replace 'target' with the actual column name

# Scale the features
scaler_audio = StandardScaler()
scaler_text = StandardScaler()
X_train_audio = scaler_audio.fit_transform(X_train_audio)
X_train_text = scaler_text.fit_transform(X_train_text)

# Model Architecture
audio_input = Input(shape=(X_train_audio.shape[1],))
audio_dense = Dense(128, activation='relu')(audio_input)

text_input = Input(shape=(X_train_text.shape[1],))
text_dense = Dense(128, activation='relu')(text_input)

merged = Concatenate()([audio_dense, text_dense])
output = Dense(1, activation='linear')(merged)

model = Model(inputs=[audio_input, text_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Model Training
history = model.fit(
    [X_train_audio, X_train_text],
    y_train,
    validation_split=0.2,
    epochs=100,
    callbacks=[
        EarlyStopping(patience=10),
        ReduceLROnPlateau(factor=0.2, patience=5)
    ]
)

# Scale the features
scaler_audio = StandardScaler()
scaler_text = StandardScaler()
X_train_audio = scaler_audio.fit_transform(X_train_audio)
X_train_text = scaler_text.fit_transform(X_train_text)
import joblib
import os

# Ensure the models/ directory exists
os.makedirs('models', exist_ok=True)

# Save the scalers
joblib.dump(scaler_audio, 'models/scaler_audio.pkl')
joblib.dump(scaler_text, 'models/scaler_text.pkl')


# Save the trained model
model.save('models/hybrid_model.h5')