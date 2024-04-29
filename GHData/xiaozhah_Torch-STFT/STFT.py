"""
code from nnAudio's STFT and has some modification
Working as GPU version of STFT implemented by SMS-tool
"""
import torch
from torch import nn
import numpy as np
from scipy import signal
import torch.nn.functional as F

def pad_center(data, size, axis=-1, **kwargs):
    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (int(size - n - lpad), lpad)

    if lpad < 0:
        raise (
            ("Target size ({:d}) must be " "at least input size ({:d})").format(size, n)
        )

    return np.pad(data, lengths, **kwargs)

## Kernal generation functions ##
def create_fourier_kernels(
    n_fft,
    win_length=None,
    sr=44100,
    window="hann",
):
    freq_bins = n_fft // 2 + 1
    n = np.arange(float(n_fft))
    wsin = np.empty((freq_bins, 1, n_fft))
    wcos = np.empty((freq_bins, 1, n_fft))
    bins2freq = []
    binslist = []

    # Choosing window shape
    window_mask = signal.get_window(window, int(win_length))
    window_mask = window_mask/np.sum(window_mask)
    window_mask = pad_center(window_mask, n_fft)

    for k in range(freq_bins):  # Only half of the bins contain useful info
        bins2freq.append(k * sr / n_fft)
        binslist.append(k)
        wsin[k, 0, :] = np.sin(
            2 * np.pi * k * n / n_fft
        )
        wcos[k, 0, :] = np.cos(
            2 * np.pi * k * n / n_fft
        )

    hM = int(n_fft)//2
    wsin_buffer = np.zeros_like(wsin)
    wsin_buffer[:, :, :hM] = wsin[:, :, -hM:]
    wsin_buffer[:, :, -hM:] = wsin[:, :, :hM]

    wcos_buffer = np.zeros_like(wcos)
    wcos_buffer[:, :, :hM] = wcos[:, :, -hM:]
    wcos_buffer[:, :, -hM:] = wcos[:, :, :hM]
    
    wsin = wsin_buffer * window_mask
    wcos = wcos_buffer * window_mask

    return (
        wsin.astype(np.float32),
        wcos.astype(np.float32),
        bins2freq,
        binslist,
    )

class STFT(nn.Module):
    def __init__(
        self,
        n_fft=2048,
        win_length=None,
        hop_length=None,
        window="hann",
        sr=22050,
        output_format="Complex",
    ):

        super().__init__()

        # Trying to make the default setting same as librosa
        if win_length == None:
            win_length = n_fft
        if hop_length == None:
            hop_length = int(win_length // 4)
        
        self.freq_bins = n_fft // 2 + 1
        self.output_format = output_format
        self.stride = hop_length
        self.n_fft = n_fft
        self.pad_amount = self.n_fft // 2
        self.window = window
        self.win_length = win_length

        # Create filter windows for stft
        (
            kernel_sin,
            kernel_cos,
            self.bins2freq,
            self.bin_list,
        ) = create_fourier_kernels(
            n_fft,
            win_length=win_length,
            window=window,
            sr=sr,
        )

        wsin = torch.tensor(kernel_sin, dtype=torch.float)
        wcos = torch.tensor(kernel_cos, dtype=torch.float)

        self.register_buffer("wsin", wsin)
        self.register_buffer("wcos", wcos)

    def forward(self, x, output_format=None):
        output_format = output_format or self.output_format
        self.num_samples = x.shape[-1]

        padding = nn.ConstantPad1d(self.pad_amount, 0)
        x = padding(x) # center is True
        # STFT
        spec_imag = -F.conv1d(x, self.wsin, stride=self.stride)
        spec_real = F.conv1d(x, self.wcos, stride=self.stride)

        assert(spec_real.shape[1] == self.freq_bins)
        assert(spec_imag.shape[1] == self.freq_bins)

        # return (B, freq_dim, T)
        if output_format == "magnitude": # magnitude spectrogram
            absX = torch.sqrt(spec_real.pow(2) + spec_imag.pow(2))
            return 20 * torch.log10(absX)
        elif output_format == "phase": # phase spectrogram (derivative)
            pX = torch.atan2(spec_imag, spec_real)
            pX = np.unwrap(pX.numpy(), axis = 1)
            pX_diff = np.diff(pX, axis = 1)
            pX_diff = torch.FloatTensor(pX_diff)
            return pX_diff
        elif output_format == "magnitude and phase":
            absX = torch.sqrt(spec_real.pow(2) + spec_imag.pow(2))
            mX = 20 * torch.log10(absX)
            pX = torch.atan2(spec_imag, spec_real)
            pX = np.unwrap(pX.numpy(), axis = 1)
            pX_diff = np.diff(pX, axis = 1)
            pX_diff = torch.FloatTensor(pX_diff)
            return mX, pX_diff

if __name__ == "__main__":
    import soundfile as sf
    y, _ = sf.read('/Users/xzhou/我的资料/coding/sms-tools-master/sounds/sine-440-490.wav') # Loading your audio
    y = torch.FloatTensor(y).reshape(1,1,-1) # casting the array into a PyTorch Tensor

    spec_layer = STFT(win_length=1024, n_fft=1024, hop_length=256, window='hanning',
                      sr=44100, output_format = "magnitude and phase")

    mX, pX_diff = spec_layer(y) # Feed-forward your waveform to get the spectrogram
    np.save('mX.npy', mX.numpy())
    np.save('pX_diff.npy', pX_diff.numpy())
