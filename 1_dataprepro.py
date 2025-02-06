import numpy as np
import matplotlib.pyplot as plt
import librosa
import noisereduce as nr
from pydub import AudioSegment, effects
import sounddevice as sd
import os 

# Define file path for a single file.
path = r'C:\Users\sahil\Desktop\Project\Extacy\AudioFiles\03-01-02-02-02-01-02.wav'

# Load the audio file into an 'AudioSegment' object, and extract the sample rate.
rawsound = AudioSegment.from_file(path)
x, sr = librosa.load(path, sr=None)

# Normalize to +5.0 dBFS
normalizedsound = effects.normalize(rawsound, headroom=5.0)
normal_x = np.array(normalizedsound.get_array_of_samples(), dtype='float32')

# Plot normalized audio
plt.figure(figsize=(12, 2))
librosa.display.waveshow(normal_x, sr=sr)
plt.title('Normalized audio')
plt.show()

# Trim silence at the beginning and end
xt, index = librosa.effects.trim(normal_x, top_db=30)

# Plot trimmed audio
plt.figure(figsize=(6, 2))
librosa.display.waveshow(xt, sr=sr)
plt.title('Trimmed audio')
plt.show()

sd.play(xt, samplerate=sr)
sd.wait()

# Right-side padding for length equalization
max_length = 173056
padded_x = np.pad(xt, (0, max_length - len(xt)), 'constant')

# Plot padded audio
plt.figure(figsize=(12, 2))
librosa.display.waveshow(padded_x, sr=sr)
plt.title('Padded audio')
plt.show()

sd.play(padded_x, samplerate=sr)
sd.wait()

# Noise reduction
final_x = nr.reduce_noise(y=padded_x, sr=sr, prop_decrease=1.0)

# Plot noise-reduced audio
plt.figure(figsize=(12, 2))
librosa.display.waveshow(final_x, sr=sr)
plt.title('Noise-reduced audio')
plt.show()

sd.play(final_x, samplerate=sr)
sd.wait()

# Feature extraction
frame_length = 2048
hop_length = 512

f1 = librosa.feature.rms(y=final_x, frame_length=frame_length, hop_length=hop_length)  # Energy - RMS
print('Energy shape:', f1.shape)
f2 = librosa.feature.zero_crossing_rate(y=final_x, frame_length=frame_length, hop_length=hop_length)  # Zero Crossing Rate
print('ZCR shape:', f2.shape)
f3 = librosa.feature.mfcc(y=final_x, sr=sr, n_mfcc=13, hop_length=hop_length)  # MFCCs
print('MFCCs shape:', f3.shape)

