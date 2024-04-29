import librosa.display
import matplotlib.pyplot as plt
import numpy as np

filename = './ESC-50/audio/1-137-A-32.wav'
y, sr = librosa.load(filename)

# trim silent edges
typing, _ = librosa.effects.trim(y)
# plot
fig, ax = plt.subplots(nrows=1)
librosa.display.waveplot(typing, sr=sr)
plt.show()

# Fourier Transform
n_fft = 2048
D = np.abs(librosa.stft(typing[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
plt.plot(D)
plt.show()

# Amplify Plot - True Spectrogram - displaying intensity of frequencies
DB = librosa.amplitude_to_db(D, ref=np.max)
librosa.display.specshow(DB, sr=sr, hop_length=n_fft+1, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.show()

# Mel Scale - transformation to place frequency on a balanced scale
n_mels = 100
mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

librosa.display.specshow(mel, sr=sr, hop_length=n_fft+1, x_axis='linear')
plt.ylabel('Mel filter')
plt.colorbar()
plt.title('1. Our filter bank for converting from Hz to mels.')
plt.show()