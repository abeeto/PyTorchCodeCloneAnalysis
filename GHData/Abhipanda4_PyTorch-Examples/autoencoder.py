import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import torchvision

import numpy as np
import matplotlib.pyplot as plt

data = torchvision.datasets.MNIST(
        root="./mnist",
        transform=torchvision.transforms.ToTensor(),
        download=True
)

# print(data.train_data.size())

# visualizee a random datapoint
# index = np.random.randint(0, 1000)
# plt.imshow(data.train_data[index].numpy(), cmap='gray')
# plt.title("Index: " + str(index) + ", label: " + str(data.train_labels[index]))
# plt.show()

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear(28 * 28, 128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, 8)
        )

        self.decoder = nn.Sequential(
                nn.Linear(8, 64),
                nn.ELU(),
                nn.Linear(64, 128),
                nn.ELU(),
                nn.Linear(128, 28 * 28),
                nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder()


# hyperparameters
N_EPOCH = 2
BATCH_SIZE = 64
N_TEST_IMG = 5
learning_rate = 0.01


optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
loss_fn = nn.SmoothL1Loss()

fig, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()

view_data = Variable(data.train_data[:N_TEST_IMG].view(-1, 28 * 28).type(torch.FloatTensor)/255)
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())


train_loader = Data.DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)
for epoch in range(N_EPOCH):
    for step, (x, y) in enumerate(train_loader):
        batch_X = Variable(x.view(-1, 28 * 28))
        batch_Y = Variable(x.view(-1, 28 * 28)) # output should be same as input

        encoded, decoded = autoencoder(batch_X)
        loss = loss_fn(decoded, batch_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print("Epoch: " + str(epoch) + ", Train Loss: " + str(loss.data[0]))

            _, decoded_data = autoencoder(view_data)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                # convert into np array and plot the output image
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            plt.draw()
            plt.pause(0.05)

plt.ioff()
plt.show()
