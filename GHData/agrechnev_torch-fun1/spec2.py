# Created by  IT-JIM  2021
# Here a neural net (of Conv1d) should try to reconstruct signal from COMPLEX spectrogram

import sys
import pathlib
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn
import torch.nn.functional
import torch.utils.data

N_DATA = 1024
WIN = 128
STEP = 64
FAKE_DSET_SIZE = 1024*8
N_EPOCHS = 50
BATCH_SIZE = 64
device = 'cuda'
# device = 'cpu'


########################################################################################################################
def print_it(a: np.ndarray, name: str = ''):
    print(name, a.shape, a.dtype, a.min(), a.mean(), a.max())


########################################################################################################################
def gen_signal1(n: int, mode: int) -> np.ndarray:
    """Generate a random 1D signal"""
    if mode == 0:
        y = np.random.randn(n)
    elif mode == 1:
        y = np.cos(np.arange(n) * 0.5) + np.cos(np.arange(n) * 1.5)
    return y


########################################################################################################################
def build_complex_spectrogram(y: np.ndarray, win: int, step: int, win_fun: int = 0) -> np.ndarray:
    """Complex Spectrogram from an 1D signal"""
    assert y.ndim == 1
    ny = len(y)
    nt = 1 + (ny - win) // step
    x = np.zeros((nt, win), dtype='float64')
    for i in range(nt):
        yy = y[i * step: i * step + win]
        fc = np.fft.rfft(yy)
        nf = len(fc)
        assert nf == win // 2 + 1
        # Real parts
        x[i, :nf] = np.real(fc)
        # Imag parts
        x[i, nf:] = np.imag(fc)[1:win - nf + 1]
    return x


########################################################################################################################
class OrcDataset1(torch.utils.data.Dataset):
    def __init__(self, sz: int, n_data: int, win: int, step: int):
        self.sz = sz
        self.n_data = n_data
        self.win = win
        self.step = step

    def __len__(self):
        return self.sz

    def __getitem__(self, item):
        # Random signal
        y = gen_signal1(self.n_data, 0)
        # SPectrogram
        x = build_complex_spectrogram(y, self.win, self.step, 0)
        if False:
            plt.subplot(1, 2, 1)
            plt.plot(y)
            plt.subplot(1, 2, 2)
            plt.imshow(x)
            plt.show()
            sys.exit()

        x = torch.tensor(x.T / np.sqrt(self.win), dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y


########################################################################################################################
class OrcModel(torch.nn.Module):
    def __init__(self, n_data, win, step):
        super(OrcModel, self).__init__()
        nt = 1 + (n_data - win) // step  # N of spectrogram timestamps
        n_levels = int(np.log2(n_data / nt))
        assert n_data % 2 ** n_levels == 0
        self.pad_to = n_data // 2 ** n_levels
        assert self.pad_to >= nt

        print(f'model: nt={nt}, n_levels={n_levels}')
        channels = [max(win // 2 ** i, 2) for i in range(n_levels + 1)]

        # Create conv and decon blocks
        n_conv = 2
        blocks = []
        for i_l in range(n_levels + 1):
            c = channels[i_l]
            # Convolution
            for i_c in range(n_conv):
                blocks.append(torch.nn.Conv1d(c, c, 3, padding=1))
                blocks.append(torch.nn.ReLU())
            # Upsample
            if i_l < n_levels:
                blocks.append(torch.nn.ConvTranspose1d(c, channels[i_l + 1], 4, stride=2, padding=1))
                blocks.append(torch.nn.ReLU())
        # Final convolution to 1 channel, no activation
        blocks.append(torch.nn.Conv1d(channels[-1], 1, 3, padding=1))
        self.net = torch.nn.Sequential(*blocks)

    def forward(self, x_in):
        x = x_in
        # Pad input data
        _, nc, nt = x.shape
        if nt < self.pad_to:
            p0 = self.pad_to - nt
            p1 = p0 // 2
            p2 = p0 - p1
            x = torch.nn.functional.pad(x, (p1, p2))
        x = self.net(x)
        n = x.shape[-1]
        x = x.view(-1, n)
        return x


########################################################################################################################
def main():
    # dataset
    dset = OrcDataset1(FAKE_DSET_SIZE, N_DATA, WIN, STEP)
    dloader = torch.utils.data.DataLoader(dset, batch_size=BATCH_SIZE)

    # Model
    model = OrcModel(N_DATA, WIN, STEP).to(device)

    if True:
        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = torch.nn.MSELoss()
        for epoch in range(N_EPOCHS):
            loss_sum = 0.0
            for x, y in dloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()
            loss_train = loss_sum / len(dloader)
            print(f'Epoch {epoch} : loss = {loss_train}')

        # Save model
        p_dir = pathlib.Path('./trained')
        if not p_dir.exists():
            p_dir.mkdir()
        torch.save(model.state_dict(), './trained/spec2.pth')

    else:
        chk = torch.load('./trained/spec2.pth')
        model.load_state_dict(chk)


########################################################################################################################
if __name__ == '__main__':
    main()
