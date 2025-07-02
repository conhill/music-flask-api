# 🎵 Music Genre Classification API

A Flask-based REST API that classifies music genres using a pre-trained Artificial Neural Network (ANN) model. The API accepts audio files and returns genre predictions based on extracted audio features using advanced music information retrieval techniques.

## 🚀 Features

- **Real-time Genre Classification**: Upload audio files and get instant genre predictions
- **RESTful API**: Clean, well-documented endpoints for easy integration
- **Advanced Feature Extraction**: Uses 26 audio features including MFCCs, spectral features, and chroma
- **Pre-trained Model**: Leverages TensorFlow/Keras neural network for accurate predictions
- **Cross-Origin Support**: CORS-enabled for frontend integration
- **Health Monitoring**: Built-in health check endpoint for monitoring

## 🛠️ Technology Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: TensorFlow/Keras for neural network inference
- **Audio Processing**: Librosa for feature extraction
- **Data Processing**: NumPy, Joblib for data handling and model serialization
- **CORS**: Flask-CORS for cross-origin requests

## 📋 Prerequisites

- Python 3.7 or higher
- pip package manager
- At least 2GB RAM for model inference
- Audio files in supported formats (WAV, MP3, FLAC, etc.)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd music-flask-api
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model files are present**
   Make sure these files exist in the project directory:
   - `model_ann.h5` - Pre-trained neural network model
   - `scaler.save` - Feature scaler for normalization
   - `label_encoder.save` - Label encoder for genre mapping

## 🚀 Running the API

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **The API will be available at:**
   - Prediction Endpoint: `http://localhost:5000/predict`

3. **Server Output:**
   ```
   Starting Music Genre Classification API...
   API Health Check: http://localhost:5000/health
   Prediction Endpoint: http://localhost:5000/predict
   * Running on http://127.0.0.1:5000
   ```

## 📖 API Documentation


### Prediction Endpoint

**POST** `/predict`

Classifies the genre of an uploaded audio file.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Audio file with key `file`

**Supported Audio Formats:**
- WAV, MP3, FLAC, M4A, OGG, and other formats supported by librosa

**Response:**
```json
{
  "prediction": "rock",
  "score": 0.85
}
```

**Error Responses:**
```json
{
  "error": "No file uploaded"
}
```

## 🎯 Usage Examples

### Using cURL

# Predict genre
curl -X POST \
  -F "file=@/path/to/your/audio.wav" \
  http://localhost:5000/predict


## 🔍 Audio Feature Extraction

The API extracts 26 audio features for classification:

1. **Chroma STFT** - Represents 12 different pitch classes
2. **RMSE** - Root Mean Square Energy (loudness measure)
3. **Spectral Centroid** - "Center of mass" of the spectrum
4. **Spectral Bandwidth** - Width of the spectrum
5. **Spectral Rolloff** - Frequency below which 85% of energy is contained
6. **Zero Crossing Rate** - Rate of sign changes in the signal
7. **MFCCs (1-20)** - Mel-frequency cepstral coefficients for timbre

## 🏗️ Model Architecture

- **Type**: Artificial Neural Network (ANN)
- **Framework**: TensorFlow/Keras
- **Input**: 26 audio features
- **Output**: Binary classification with confidence score
- **Preprocessing**: StandardScaler for feature normalization

## 🐛 Troubleshooting

### Common Issues

1. **Model files not found**
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'model_ann.h5'
   ```
   - Ensure all model files (`model_ann.h5`, `scaler.save`, `label_encoder.save`) are in the project directory

2. **Audio processing errors**
   ```
   librosa.util.exceptions.ParameterError
   ```
   - Check if the audio file is corrupted or in an unsupported format
   - Try converting to WAV format

3. **Memory issues**
   ```
   MemoryError: Unable to allocate memory
   ```
   - Ensure sufficient RAM is available
   - Consider processing shorter audio clips

### Debug Mode

The API runs in debug mode by default. For production deployment:

```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

## 🚀 Production Deployment

For production deployment, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn --bind 0.0.0.0:5000 app:app
```

## 📊 Performance

- **Processing Time**: ~1-3 seconds per audio file
- **Memory Usage**: ~500MB-1GB depending on model size
- **Supported Audio Length**: Up to 10 seconds (configurable)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Librosa** - For excellent audio processing capabilities
- **TensorFlow/Keras** - For deep learning framework
- **Flask** - For the lightweight web framework
- **Music Information Retrieval Community** - For research and best practices

## 📧 Contact

For questions or support, please open an issue in the GitHub repository.

---

⭐ **Star this repository if you found it helpful!**
