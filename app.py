"""
Music Genre Classification API

A Flask-based REST API that classifies music genres using a pre-trained 
Artificial Neural Network (ANN) model. The API accepts audio files and 
returns genre predictions based on extracted audio features.

Dependencies:
- Flask: Web framework for the API
- TensorFlow: Deep learning framework for model inference
- Librosa: Audio analysis library for feature extraction
- Joblib: For loading pre-trained preprocessing components
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import librosa
import joblib
import os

# Initialize Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS) to allow frontend requests
CORS(app)

# Load pre-trained models and preprocessing components
# These files should be generated during the training phase
model = tf.keras.models.load_model('model_ann.h5')  # Pre-trained ANN model
scaler = joblib.load('scaler.save')                  # Feature scaler for normalization
encoder = joblib.load('label_encoder.save')         # Label encoder for genre mapping

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint to verify API is running
    
    Returns:
        JSON response with status and loaded model information
    """
    return jsonify({
        'status': 'healthy',
        'message': 'Music Genre Classification API is running',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'encoder_loaded': encoder is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for predicting music genre from audio file
    
    Accepts POST requests with audio file uploads and returns genre classification
    along with confidence score.
    
    Returns:
        JSON response with:
        - prediction: The predicted genre label
        - score: Confidence score (0-1)
        
    Error codes:
        400: No file uploaded or invalid file format
        500: Internal server error during processing
    """
    # Check if file was uploaded in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    temp_path = 'temp.wav'
    
    try:
        # Save uploaded file temporarily for processing
        file.save(temp_path)
        
        # Load audio file with librosa
        # Parameters: mono=True converts to single channel, duration=10 limits to 10 seconds
        y, sr = librosa.load(temp_path, mono=True, duration=10)
        
        # Extract audio features that match the training pipeline
        # These 26 features are commonly used in music information retrieval
        
        # 1. Chroma STFT: Represents the 12 different pitch classes
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        
        # 2. Root Mean Square Energy: Measure of audio power/loudness
        rmse = np.mean(librosa.feature.rms(y=y))
        
        # 3. Spectral Centroid: "Center of mass" of the spectrum (brightness)
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # 4. Spectral Bandwidth: Width of the spectrum
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        
        # 5. Spectral Rolloff: Frequency below which 85% of energy is contained
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # 6. Zero Crossing Rate: Rate of sign changes in the signal
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # 7-26. MFCCs (Mel-frequency cepstral coefficients): 
        # Capture the shape of the spectral envelope, important for timbre
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfccs = [np.mean(mfcc[i]) for i in range(20)]  # First 20 MFCC coefficients
        
        # Combine all features into a single array
        # Shape: (1, 26) - one sample with 26 features
        features = np.array([[chroma_stft, rmse, spec_cent, spec_bw, rolloff, zcr, *mfccs]])
        
        # Normalize features using the same scaler from training
        # This ensures features are on the same scale as training data
        features_scaled = scaler.transform(features)
        
        # Make prediction using the trained neural network
        # Returns probability array, we take the first (and only) prediction
        pred = model.predict(features_scaled)[0][0]
        
        # Convert probability to binary classification using 0.5 threshold
        # Then decode the numeric label back to genre name
        label = encoder.inverse_transform([int(pred > 0.5)])[0]
        
        # Clean up temporary file
        os.remove(temp_path)
        
        # Return prediction results as JSON
        return jsonify({
            'prediction': label, 
            'score': float(pred)
        })
        
    except Exception as e:
        # Handle any errors during processing
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Start the Flask development server
    # In production, use a WSGI server like Gunicorn instead
    print("Starting Music Genre Classification API...")
    print("API Health Check: http://localhost:5000/health")
    print("Prediction Endpoint: http://localhost:5000/predict")
    app.run(debug=True, port=5000)