from librosa import load, piptrack
import numpy as np
from scipy.signal.windows import blackmanharris
import torch


def load_file(filename):
    y, sr = load(filename)
    pitches, magnitudes = piptrack(y=y, sr=sr, fmin=50, fmax=400)

    q25 = 0.0
    iqr = 0.0
    freq = 0.0

    if len(pitches[np.nonzero(pitches)]) > 0:
        q25 = np.percentile(pitches[np.nonzero(pitches)], 25)
        iqr = np.percentile(pitches[np.nonzero(pitches)], 75) - q25

        windowed = y * blackmanharris(len(y))
        median = np.argmax(abs(np.fft.rfft(windowed)))

        freq = median / len(windowed)

    out = torch.tensor([q25, iqr, freq])

    return out

