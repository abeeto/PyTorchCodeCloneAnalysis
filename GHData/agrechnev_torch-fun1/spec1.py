# By IT-JIM, 2021
# Here Pytorch will learn inverse FFT: should be trivial

import sys
import numpy as np
import torch
import torch.utils.data

N_DATA = 128
FAKE_DSET_SIZE = 1024 * 16
N_EPOCHS = 10
device = 'cuda'


########################################################################################################################
def print_it(a: np.ndarray, name: str = ''):
    print(name, a.shape, a.dtype, a.min(), a.mean(), a.max())


########################################################################################################################
class GoblinDset(torch.utils.data.Dataset):
    """Note indexing here is a pure fake, __getitem__() returns random data each time !"""

    def __init__(self, n, sz):
        self.n = n
        self.sz = sz

    def __len__(self):
        return self.sz

    def __getitem__(self, item):
        """item is ignored, returns random data"""
        y = np.random.randn(self.n)
        f = np.fft.rfft(y)
        x = np.concatenate([f.real, f.imag[1:-1]])
        x = torch.tensor(x, dtype=torch.float32) / np.sqrt(self.n)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y


########################################################################################################################
def main():
    dset = GoblinDset(N_DATA, FAKE_DSET_SIZE)
    loader = torch.utils.data.DataLoader(dset, batch_size=64)
    print('len(dset) = ', len(dset))
    print('len(loader) = ', len(loader))
    net = torch.nn.Linear(N_DATA, N_DATA).to(device)
    optimizer = torch.optim.Adam(net.parameters())
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

    print('SINGLE PREDICTION !')
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        out = net(x)

    print('OUT : ', out[11, :5])
    print('Y : ', y[11, :5])


########################################################################################################################
if __name__ == '__main__':
    main()
