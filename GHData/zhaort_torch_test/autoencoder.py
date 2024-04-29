import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

torch.manual_seed(1)

EPOCH = 10
BATCH_SIZE = 32
LR = 0.0001
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

# train_data = torchvision.datasets.MNIST(root='./mnist/', train=True,
#                                         transform=torchvision.transforms.ToTensor(),
#                                         download=DOWNLOAD_MNIST)
data_set = datasets.ImageFolder(root=r'D:\BaiduNetdiskDownload\faces',
                                transform=transforms.Compose([
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor()])
                                )
train_loader = Data.DataLoader(dataset=data_set, batch_size=BATCH_SIZE, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3*128*128, 1280),
            nn.Tanh(),
            nn.Linear(1280, 640),
            nn.Tanh(),
            nn.Linear(640, 120),
            nn.Tanh(),
            nn.Linear(120, 30),
            nn.Tanh(),
            nn.Linear(30, 10),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 30),
            nn.Tanh(),
            nn.Linear(30, 120),
            nn.Tanh(),
            nn.Linear(120, 640),
            nn.Tanh(),
            nn.Linear(640, 1280),
            nn.Tanh(),
            nn.Linear(1280, 3*128*128),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


auto_encoder = AutoEncoder()
auto_encoder.load_state_dict(torch.load('auto_encoder'))
optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=LR)
loss_fn = nn.MSELoss()

for epoch in range(EPOCH):
    for step, (x, _) in enumerate(train_loader):
        x = x.view(-1, 3*128*128)
        y = x.view(-1, 3*128*128)
        label = y

        encoded, decoded = auto_encoder(x)

        loss = loss_fn(decoded, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 and step != 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())
            torch.save(auto_encoder.state_dict(), 'auto_encoder')
            if step % 1000 == 0:
                torchvision.transforms.ToPILImage()(decoded[0].reshape(3, 128, 128)).resize((128, 128)).show()

torch.save(auto_encoder.state_dict(), 'auto_encoder')
