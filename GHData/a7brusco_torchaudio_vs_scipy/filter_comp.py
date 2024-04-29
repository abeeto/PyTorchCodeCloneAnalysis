import sys
import torch
import torchaudio.functional
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

def getSweep(freq_min: float=20.0, freq_max: float=20000.0, sample_rate:int=44100, length:int=220500, length_fade:int=0, amplitude_signal_final:float=0.2):
    T = length / sample_rate
    K = 2.0 * np.pi * freq_min * T / np.log(freq_max / freq_min)
    i = np.arange(length)
    L = i / length * np.log(freq_max / freq_min)
    samples = amplitude_signal_final * np.sin(K * (np.exp(L) - 1.0))
    for i in range(length_fade):
        coeff = np.float64(i) * np.float64(i) / (np.float64(length_fade) * np.float64(length_fade))
        samples[i] *= coeff
        samples[length - i - 1] *= coeff
    return samples

def get_coeffs(tensor:bool = False) -> dict:
    a_coeffs = [1.0,-4.019576169118377,6.18940639545866,-4.453198832384563,1.4208429009515808,-0.14182546010746952,0.004351176485980078]
    b_coeffs = [0.2557411285359176,-0.5114822570718351,-0.25574112853591763,1.02296451414367,-0.2557411285359177,-0.5114822570718351,0.2557411285359176]
    if tensor:
        return {'denom': torch.FloatTensor(a_coeffs), 'num': torch.FloatTensor(b_coeffs)}
    else:
        return {'denom': a_coeffs, 'num': b_coeffs}

def apply_torch_filter(s_in: torch.Tensor) -> torch.Tensor:
    coeffs = get_coeffs(True)
    filt_output = torchaudio.functional.lfilter(s_in, coeffs['denom'], coeffs['num'], clamp=False)
    return filt_output

def apply_scipy_filter(s_in: np.ndarray) -> np.ndarray:
    coeffs = get_coeffs()
    s_out = scipy.signal.lfilter(coeffs['num'], coeffs['denom'], s_in.flatten())
    return s_out

if __name__ == "__main__":
    sweep = getSweep()
    sweep_tensor = torch.from_numpy(sweep.astype('float32'))
    scipy_output = apply_scipy_filter(sweep_tensor.numpy())
    torch_output = apply_torch_filter(sweep_tensor)

    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(scipy_output, label="scipy")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(torch_output.numpy(), label="pytorch")
    plt.legend()
    plt.show()
