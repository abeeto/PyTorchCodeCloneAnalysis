import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as Data
import torchvision

data_set = torchvision.datasets.MNIST(root='./data/mnist_data', train=True, transform=torchvision.transforms.ToTensor(), download=False)
data_loader = Data.DataLoader(dataset=data_set, batch_size=64, shuffle=True)

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 3),
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

autoencoder = AutoEncoder()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)

for epoch in range(3):
    for steps, (x, y) in enumerate(data_loader):
        b_x = x.view(-1, 28*28)
        b_y = x.view(-1, 28*28)

        encode, decode = autoencoder(b_x)

        loss = criterion(decode, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if steps % 100 == 0:
            print('Epoch: {} | step: {} | Loss: {:.4f}'.format(epoch, steps, loss.data.numpy()))
