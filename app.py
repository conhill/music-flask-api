from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import librosa
import joblib
import os

app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model('model_ann.h5')
scaler = joblib.load('scaler.save')  # Make sure you saved this after training
encoder = joblib.load('label_encoder.save')  # Make sure you saved this after training

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    temp_path = 'temp.wav'
    file.save(temp_path)
    # Extract features (must match your training pipeline)
    y, sr = librosa.load(temp_path, mono=True, duration=10)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    rmse = np.mean(librosa.feature.rms(y=y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfccs = [np.mean(mfcc[i]) for i in range(20)]
    features = np.array([[chroma_stft, rmse, spec_cent, spec_bw, rolloff, zcr, *mfccs]])
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0][0]
    label = encoder.inverse_transform([int(pred > 0.5)])[0]
    os.remove(temp_path)
    return jsonify({'prediction': label, 'score': float(pred)})

if __name__ == '__main__':
    app.run(port=5000)