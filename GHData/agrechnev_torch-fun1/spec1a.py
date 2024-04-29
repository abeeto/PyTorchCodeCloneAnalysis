# Created by  IT-JIM  2021
# Here I try to learn inverse FFT with 1d convolutions. Will it work?

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data

N_LEVELS = 4
N_DATA = 2**N_LEVELS
FAKE_DSET_SIZE = 1024 * 16
N_EPOCHS = 100
BATCH_SIZE = 64
NO_FOURIER = True
device = 'cuda'


########################################################################################################################
def print_it(a: np.ndarray, name: str = ''):
    print(name, a.shape, a.dtype, a.min(), a.mean(), a.max())


########################################################################################################################
class GoblinDsetA(torch.utils.data.Dataset):
    """Note indexing here is a pure fake, __getitem__() returns random data each time !"""

    def __init__(self, n, sz):
        self.n = n
        self.sz = sz

    def __len__(self):
        return self.sz

    def __getitem__(self, item):
        """item is ignored, returns random data"""
        y = np.random.randn(self.n)
        if NO_FOURIER:
            x = y * np.sqrt(self.n)
        else:
            f = np.fft.rfft(y)
            x = np.concatenate([f.real, f.imag[1:-1]])
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1) / np.sqrt(self.n)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        return x, y


########################################################################################################################
class StupidNet(torch.nn.Module):
    def __init__(self):
        super(StupidNet, self).__init__()
        n_conv = 1
        channels = 2 ** np.arange(N_LEVELS+1)[::-1]
        channels[1:] *= 2
        print('channels=', channels)
        # channels[-1] = 2
        blocks = []
        for i_l in range(N_LEVELS):
            c = channels[i_l]
            # Convolution
            if i_l > 0:
                for i_c in range(n_conv):
                    blocks.append(torch.nn.Conv1d(c, c, 3, padding=1))
                    blocks.append(torch.nn.ReLU())
            # Upsample
            blocks.append(torch.nn.ConvTranspose1d(c, channels[i_l + 1], 4, stride=2, padding=1))
            blocks.append(torch.nn.ReLU())
        # Final convolution to 1 channel
        blocks.append(torch.nn.Conv1d(channels[-1], 1, 3, padding=1))
        self.net = torch.nn.Sequential(*blocks)

    def forward(self, x_in):
        return self.net(x_in)


########################################################################################################################
def main():
    dset = GoblinDsetA(N_DATA, FAKE_DSET_SIZE)
    loader = torch.utils.data.DataLoader(dset, batch_size=BATCH_SIZE)
    print('len(dset) = ', len(dset))
    print('len(loader) = ', len(loader))
    net = StupidNet().to(device)
    print('net=', net)

    if False:
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        print_it(x, 'x')
        print_it(y, 'y')
        out = net(x)
        print_it(out, 'out')
        sys.exit(0)


    if True:
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        for epoch in range(N_EPOCHS):
            loss_sum = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = net(x)
                loss = criterion(out, y)
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()
            loss_train = loss_sum / len(loader)
            print(f'Epoch {epoch} : loss = {loss_train}')

        # Save model
        p_dir = pathlib.Path('./trained')
        if not p_dir.exists():
            p_dir.mkdir()
        torch.save(net.state_dict(), './trained/spec1a.pth')
    else:
        chk = torch.load('./trained/spec1a.pth')
        net.load_state_dict(chk)

    print('SINGLE PREDICTION !')
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        out = net(x)

    print('SINGLE PREDICTION (NOT RANDOM)')
    y = np.sin(np.arange(16) * np.pi / 16)
    if NO_FOURIER:
        x = y * np.sqrt(N_DATA)
    else:
        f = np.fft.rfft(y)
        x = np.concatenate([f.real, f.imag[1:-1]])
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1) / np.sqrt(N_DATA)
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        out = net(x)
    out = out.detach().cpu().squeeze().numpy()
    plt.plot(y)
    plt.plot(out)
    plt.show()

    print('OUT : ', out[11, 0, :10])
    print('Y : ', y[11, 0, :10])


########################################################################################################################
if __name__ == '__main__':
    main()
