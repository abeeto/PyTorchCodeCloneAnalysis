import torch
from math import pi

# input shape: NxB
# output shape: NxTx(1+harmonics)
class SineGenerator(torch.nn.Module):
    def __init__(self, waveform_length):
        super(SineGenerator, self).__init__()
        self.waveform_length = waveform_length # T
        self.sr = 16000 # sampling rate
        self.F0_mag = 0.1 # magnitude of the sine waves
        self.noise_mag = 0.003 # magnitude of noise
        self.bins = 64 # the number of random initial phase
        self.harmonics = 7 # number of harmonics
        self.natural_waveforms = None

    def forward(self, x, y=None):
        # interpolate x (from NxBx1 to NxTx1)
        f = torch.nn.functional.interpolate(
            x.unsqueeze(-1).transpose(1, 2), self.waveform_length
        ).transpose(1, 2)
        h = torch.arange(1, self.harmonics + 2).unsqueeze(0).unsqueeze(0)
        n = torch.normal(
            0, self.noise_mag**2, (self.waveform_length,)
        ).unsqueeze(-1)

        if self.natural_waveforms is not None and y is None:
            y = self.natural_waveforms
        if y is not None:
            # generate candidates of initial phase
            phis = torch.linspace(-pi, pi, self.bins)
            # calculate the cross correlation for each initial phase
            voiced = self.F0_mag * torch.sin(
                2 * pi * torch.cumsum(f, 1) / self.sr + phis
            ) + n
            unvoiced = 1. / (3 * self.noise_mag) * n
            signals = torch.where(f > 0, voiced, unvoiced)
            phi_idx = torch.argmax(torch.sum(
                y.unsqueeze(-1) * signals, 1
            ), 1)
            phi = phis[phi_idx]
        else:
            phi = (torch.rand(x.size(0)) - 0.5) * 2 * pi
        phi = phi.unsqueeze(-1).unsqueeze(-1)
        voiced = self.F0_mag * torch.sin(
            h * 2 * pi * torch.cumsum(f, 1) / self.sr + phi
        ) + n
        unvoiced = 1. / (3 * self.noise_mag) * n
        return torch.where(f > 0, voiced, unvoiced)

# input: NxTx(context_dim+input_dim)
# output: NxTxoutput_dim
class WaveNetCore(torch.nn.Module):
    def __init__(self, context_dim, output_dim):
        super(WaveNetCore, self).__init__()
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.weight = torch.nn.Parameter(
            torch.randn(context_dim, 2 * output_dim)
        )

    def forward(self, x, c):
        weight_context = torch.matmul(c, self.weight)
        h = x + weight_context
        h1 = h[:, :, :self.output_dim]
        h2 = h[:, :, self.output_dim:]
        return torch.tanh(h1) * torch.sigmoid(h2)
