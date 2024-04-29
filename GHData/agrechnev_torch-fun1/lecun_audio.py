# By IT-JIM, 22-Jan-2021
import sys

import librosa
import librosa.display
import sounddevice as sd

import numpy as np
import matplotlib.pyplot as plt

import lecun_plot


########################################################################################################################
def main():
    lecun_plot.set_default()
    x, sampling_rate = librosa.load('data/win_xp_shutdown.wav')
    n = len(x)
    len_t = n / sampling_rate
    print(f'{n} samples, sr={sampling_rate}, {len_t} seconds !')
    t = np.arange(n) / sampling_rate

    if False:
        sd.play(x, sampling_rate)
        sd.wait()

    if False:
        plt.figure()
        plt.plot(t, x)
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title(r'$x(t)$')
        plt.show()

    # Now the spectrogram ?
    xf = librosa.stft(x)
    # print(f'xf : {xf.shape}, {xf.dtype}')
    xf_db = librosa.amplitude_to_db(np.abs(xf))
    # print(f'xf_db : {xf_db.shape}, {xf_db.dtype}')

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.xlim([0, len_t])
    plt.ylabel('amplitude')
    plt.title('Signal x(t) and its spectrogram')
    plt.subplot(2, 1, 2)
    librosa.display.specshow(xf_db, sr=sampling_rate, x_axis='time', y_axis='hz')
    plt.ylim(top=2000)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.grid(True)
    plt.show()


########################################################################################################################
if __name__ == '__main__':
    main()