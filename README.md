# Sentimental
The Speech Emotion Recognition (SER) is an important topic of research that uses speech patterns to infer a speaker's emotional state. 
Emotion Recognition from Speech Audio Files
This project focuses on identifying emotions from speech audio files using various audio processing techniques and a Long Short-Term Memory (LSTM) deep learning model. The system preprocesses raw audio, extracts key speech features, and predicts the underlying emotion using a trained LSTM model.

Table of Contents
Project Overview
Dataset
Audio Preprocessing
Feature Extraction
Model Architecture
Installation
Usage
Model Training
Real-Time Emotion Prediction
Results and Visualization
File Descriptions
References
Project Overview
This project applies audio signal processing techniques combined with deep learning to recognize emotions from audio files. The primary components of the system include:

Preprocessing: Normalize, trim, and denoise raw audio files.
Feature Extraction: Extract Mel Frequency Cepstral Coefficients (MFCC), Root Mean Square Energy (RMS), and Zero-Crossing Rate (ZCR) from audio.
Deep Learning Model: Use an LSTM-based neural network for classification.
Real-Time Emotion Detection: The system can predict emotions from real-time audio input and visualize emotion probabilities.
Dataset
The system uses audio files from the Toronto Emotional Speech Set (TESS) dataset for training. Each file in the dataset is labeled with a corresponding emotion (e.g., happy, sad, angry).

Audio Preprocessing
Normalization: Audio signals are normalized to a +5.0 dBFS level to maintain uniform volume levels.
Trimming: Silence at the beginning and end of each audio file is trimmed.
Padding: Audio files are padded to a fixed length to ensure consistency across all samples.
Noise Reduction: Noise is reduced using a noise reduction algorithm to enhance audio quality.
Feature Extraction
The following features are extracted from each preprocessed audio file:

MFCC (Mel Frequency Cepstral Coefficients): Captures spectral features of the audio signal.
RMS (Root Mean Square Energy): Measures the energy of the audio signal.
ZCR (Zero Crossing Rate): Measures the rate at which the signal changes sign.
These features are used to train the deep learning model.

Model Architecture
The model used is a Sequential LSTM-based neural network. It consists of:

Two LSTM layers with 64 units each.
A dense output layer with 8 units (corresponding to 8 emotions), using softmax activation.
The model is trained to classify emotions into one of the following categories:

Neutral
Calm
Happy
Sad
Angry
Fear
Disgust
Surprise
Installation
Requirements
Install the required Python libraries:
bash
Copy code
pip install -r requirements.txt
Libraries Used:
numpy
librosa
pydub
tensorflow
keras
noisereduce
matplotlib
sounddevice
pyaudio
seaborn
sklearn
json_tricks
Usage
Preprocessing and Feature Extraction
The script loads and preprocesses audio files from a directory, extracts features, and saves them in JSON format for training the model.

Model Training
To train the model, run the training script which processes the features extracted from the audio files and fits them to the LSTM model.

bash
Copy code
python train_model.py
The model and weights are saved in JSON and HDF5 formats respectively.

Real-Time Emotion Prediction
The script allows you to capture audio from a microphone and predicts the emotion in real-time:

bash
Copy code
python realtime_emotion_recognition.py
This will open an audio stream, record a sample, and display a bar chart of the predicted emotion probabilities.

Results and Visualization
During training, the model's performance is monitored using validation accuracy and loss. Results are visualized using matplotlib, showing training loss, accuracy, and a confusion matrix for validation predictions.

File Descriptions
train_model.py: Trains the LSTM model using the preprocessed dataset and saves the model and weights.
realtime_emotion_recognition.py: Uses the trained model for real-time emotion prediction from a microphone input.
preprocess.py: Handles audio preprocessing (trimming, normalization, denoising) and feature extraction.
model8723.json: The saved model architecture in JSON format.
model8723_weights.h5: The saved model weights in HDF5 format.
requirements.txt: Python package dependencies for the project.
References
TESS Dataset: https://tspace.library.utoronto.ca/handle/1807/24487
Librosa: https://librosa.github.io/librosa/
TensorFlow/Keras: https://www.tensorflow.org/
Pydub: https://github.com/jiaaro/pydub
