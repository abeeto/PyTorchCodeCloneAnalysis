import scipy.signal as signal
import numpy as np

import librosa
import copy

from params import param

def spectrogram2wav(mag):
    '''
    input:
        mag : magnitude spectrogram (n_mag, T)
    spectrogram -> wav
    '''
    mag = mag.cpu().numpy()
    mag = mag.T # (T, n_mag)

    # de-normalize
    mag = (np.clip(mag, 0, 1)* param.max_db)- param.max_db + param.ref_db

    # dB to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag**param.power)

    # de-emphasis 
    wav = signal.lfilter([1], [1, -param.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def griffin_lim(spectrogram):
    X_best = copy.deepcopy(spectrogram)
    for i in range(param.gl_iter):
        X_t = librosa.istft(X_best, param.hopSize,
                            win_length=param.winSize, window="hann")
        est = librosa.stft(X_t, param.n_fft, param.hopSize, win_length=param.winSize)
        phase = est/np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase

    X_t = librosa.istft(X_best, param.hopSize, win_length=param.winSize, window="hann")
    y = np.real(X_t)

    return y
        