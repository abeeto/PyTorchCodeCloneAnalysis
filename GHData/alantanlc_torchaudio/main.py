# PyTorch is an open source deep learning platform that provides a seamless path from research prototyping to production deployment with GPU support.

# Significant effort in solving machine learning problems goes into data preparatio. torchaudio leverages PyTorch's GPU support, and provides many tools to make data loading easy and more readable. In this tutorial, we will see how to load and preprocess data from a simple dataset

# For this tutorial, please make sure the matplotlib package is installed for easier visualization

import torch
import torchaudio
import matplotlib.pyplot as plt

# Opening a dataset
# torchaudio supports loading sound files in the wav and mp3 format. We call waveform the resulting raw audio signal.
filename = 'data/Ses01F_impro01_M002.wav'
waveform, sample_rate = torchaudio.load(filename)

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.figure()
plt.plot(waveform.t().numpy())

# Transformations
# torchaudio supports a growing list of transformations
# 1. Resample: Resample waveform to a different sample rate.
# 2. Spectrogram: Create a spectrogram from a waveform.
# 3. MelScale: This turns a normal STFT into a Mel-frequency STFT, using a conversion matrix.
# 4. AmplitudeToDB: This turns a spectrogram from the power/amplitude scale to the decibel scale.
# 5. MFCC: Create the Mel-frequency cepstrum coefficients from a waveform.
# 6. MelSpectrogram: Create MEL Spectrograms from a waveform using the STFT function in PyTorch.
# 7. MuLawEncoding: Encode waveform based on mu-law companding.
# 8. MuLawDecoding: Decode mu-law encoded waveform.

# Since all transforms are nn.Modules or jit.ScriptModules, they can be used as part of a neural network at any point.
# To start, we can look at the log of the spectrogram on a log scale.
specgram = torchaudio.transforms.Spectrogram()(waveform)
print("Shape of spectrogram: {}".format(specgram.size()))
plt.figure()
plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='gray')

# Or we can look at the Mel Spectrogram on a log scale.
melspecgram = torchaudio.transforms.MelSpectrogram()(waveform)
print('Shape of spectrogram: {}'.format(specgram.size()))
plt.figure()
p = plt.imshow(melspecgram.log2()[0,:,:].detach().numpy(), cmap='gray')

plt.show()
print("End of program")