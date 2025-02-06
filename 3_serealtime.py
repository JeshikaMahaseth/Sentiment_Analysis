import os
from json_tricks import load

import numpy as np

import librosa
from pydub import AudioSegment, effects
import noisereduce as nr

import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
from keras.models import load_model

import matplotlib.pyplot as plt

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer




# Paths to model architecture and weights
saved_model_path = r'C:/Users/sahil/Desktop/Project/Extacy/model8723.json'
saved_weights_path = r'C:/Users/sahil/Desktop/Project/Extacy/model8723_weights.h5'

# Check if the weights file exists
if os.path.exists(saved_weights_path):
    print("Weights file found.")
else:
    print("Weights file not found. Please check the path.")

# Recreate the model architecture
model = Sequential()
model.add(InputLayer(batch_input_shape=(None, 339, 15), name='lstm_input'))
model.add(LSTM(64, return_sequences=True, name='lstm'))
model.add(LSTM(64, name='lstm_1'))
model.add(Dense(8, activation='softmax', name='dense'))

# Load the weights if the file exists
if os.path.exists(saved_weights_path):
    model.load_weights(saved_weights_path)
    print("Weights loaded successfully.")
else:
    print("Could not load weights. File not found.")

# Compiling the model
model.compile(loss='categorical_crossentropy',
              optimizer='RMSprop',
              metrics=['categorical_accuracy'])

# Print model summary
print(model.summary())

def preprocess(file_path, frame_length = 2048, hop_length = 512):
    '''
    A process to an audio .wav file before execcuting a prediction.
      Arguments:
      - file_path - The system path to the audio file.
      - frame_length - Length of the frame over which to compute the speech features. default: 2048
      - hop_length - Number of samples to advance for each frame. default: 512

      Return:
        'X_3D' variable, containing a shape of: (batch, timesteps, feature) for a single file (batch = 1).
    '''
    # Fetch sample rate.
    _, sr = librosa.load(path = file_path, sr = None)
    # Load audio file
    rawsound = AudioSegment.from_file(file_path, duration = None)
    # Normalize to 5 dBFS
    normalizedsound = effects.normalize(rawsound, headroom = 5.0)
    # Transform the audio file to np.array of samples
    normal_x = np.array(normalizedsound.get_array_of_samples(), dtype = 'float32')
    # Noise reduction
    final_x = nr.reduce_noise(normal_x, sr=sr)


    f1 = librosa.feature.rms(y=final_x, frame_length=frame_length, hop_length=hop_length, center=True, pad_mode='reflect').T # Energy - Root Mean Square
    f2 = librosa.feature.zero_crossing_rate(final_x, frame_length=frame_length, hop_length=hop_length,center=True).T # ZCR
    f3 = librosa.feature.mfcc(y=final_x, sr=sr, S=None, n_mfcc=13, hop_length = hop_length).T # MFCC
    X = np.concatenate((f1, f2, f3), axis = 1)

    X_3D = np.expand_dims(X, axis=0)

    return X_3D

# Emotions list is created for a readable form of the model prediction.

emotions = {
    0 : 'neutral',
    1 : 'calm',
    2 : 'happy',
    3 : 'sad',
    4 : 'angry',
    5 : 'fear',
    6 : 'disgust',
    7 : 'surprise',
}
emo_list = list(emotions.values())

def is_silent(data):
    # Returns 'True' if below the 'silent' threshold
    return max(data) < 100


import pyaudio
import wave
from array import array
import struct
import time

# Initialize variables
RATE = 24414
CHUNK = 512
RECORD_SECONDS = 7.1

FORMAT = pyaudio.paInt32
CHANNELS = 1
WAVE_OUTPUT_FILE = r"C:\Users\sahil\Desktop\Project\Extacy\output.wav"

import pyaudio

p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))


# Initialize a non-silent signals array to state "True" in the first 'while' iteration.
data = array('h', np.random.randint(size = 512, low = 0, high = 500))



# Open an input channel
p = pyaudio.PyAudio()
stream = None
try:
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
except OSError as e:
    print(f"Error opening stream: {e}")
    exit(1)

if stream is None:
    print("Failed to create stream object")
    exit(1)

# ... rest of your code ...

# SESSION START
print("** session started")
total_predictions = []  # A list for all predictions in the session.
tic = time.perf_counter()

while is_silent(data) == False:
    print("* recording...")
    frames = []
    data = np.nan  # Reset 'data' variable.

    timesteps = int(RATE / CHUNK * RECORD_SECONDS)  # => 339

    # Insert frames to 'output.wav'.
    if stream is not None:
        for i in range(0, timesteps):
            data = array('l', stream.read(CHUNK))
            frames.append(data)

            wf = wave.open(WAVE_OUTPUT_FILE, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
    else:
        print("Stream object is None, cannot record audio")
        break

    # ... rest of your code ...


    print("* done recording")

    x = preprocess(WAVE_OUTPUT_FILE) # 'output.wav' file preprocessing.
    # Model's prediction => an 8 emotion probabilities array.
    predictions = model.predict(x)
    pred_list = list(predictions)
    pred_np = np.squeeze(np.array(pred_list).tolist(), axis=0) # Get rid of 'array' & 'dtype' statments.
    total_predictions.append(pred_np)

    # Present emotion distribution for a sequence (7.1 secs).
    fig = plt.figure(figsize = (10, 2))
    plt.bar(emo_list, pred_np, color = 'darkturquoise')
    plt.ylabel("Probabilty (%)")
    plt.show()

    max_emo = np.argmax(predictions)
    print('max emotion:', emotions.get(max_emo,-1))

    print(100*'-')

    # Define the last 2 seconds sequence.
    last_frames = np.array(struct.unpack(str(96 * CHUNK) + 'B' , np.stack(( frames[-1], frames[-2], frames[-3], frames[-4],
                                                                            frames[-5], frames[-6], frames[-7], frames[-8],
                                                                            frames[-9], frames[-10], frames[-11], frames[-12],
                                                                            frames[-13], frames[-14], frames[-15], frames[-16],
                                                                            frames[-17], frames[-18], frames[-19], frames[-20],
                                                                            frames[-21], frames[-22], frames[-23], frames[-24]),
                                                                            axis =0)) , dtype = 'b')
    if is_silent(last_frames): # If the last 2 seconds are silent, end the session.
        break

# SESSION END
toc = time.perf_counter()
if stream is not None:
    stream.stop_stream()
    stream.close()
p.terminate()
if 'wf' in locals() and wf is not None:
    wf.close()
print('** session ended')

# Present emotion distribution for the whole session.
total_predictions_np =  np.mean(np.array(total_predictions).tolist(), axis=0)
fig = plt.figure(figsize = (10, 5))
plt.bar(emo_list, total_predictions_np, color = 'indigo')
plt.ylabel("Mean probabilty (%)")
plt.title("Session Summary")
plt.show()

print(f"Emotions analyzed for: {(toc - tic):0.4f} seconds")