import torch
from torchvision.datasets import MNIST
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
import torch.nn as nn

from sklearn.metrics import accuracy_score

device = torch.device('cpu')
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

data_tfs = tfs.Compose([
  tfs.ToTensor(),
  tfs.Normalize((0.5), (0.5))
])

# install for train and test
root = './'
train = MNIST(root, train=True,  transform=data_tfs, download=True)
test  = MNIST(root, train=False, transform=data_tfs, download=True)
#
# # print(f'Data size:\n\t train {len(train)},\n\t test {len(test)}')
# # print(f'Data shape:\n\t features {train[0][0].shape},\n\t target {type(test[0][1])}')
#
batch_size = 128
#
train_loader = DataLoader(train, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, drop_last=True)

features = 784
classes = 10

model = nn.Sequential()
model.add_module('0', nn.Linear(features, 64))
model.add_module('1', nn.Linear(64, classes))

# print(model)

criterion = nn.CrossEntropyLoss()      # (logsoftmax + negative likelihood) in its core, applied to logits
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

epochs = 3
history = []

for i in range(epochs):
  for x_batch, y_batch in train_loader:
    # 1. load batches of data correctly
    x_batch = x_batch.view(x_batch.shape[0], -1).to(device)
    y_batch = y_batch.to(device)

    # 2. compute scores with .forward or .__call__
    logits = model(x_batch)

    # 3. compute loss
    loss = criterion(logits, y_batch)
    history.append(loss.item())

    # 4. calc gradients
    optimizer.zero_grad()
    loss.backward()

    # 5. step of gradient descent
    optimizer.step()

  print(f'{i+1},\t loss: {history[-1]:.9}')



  acc = 0
  batches = 0

  for x_batch, y_batch in test_loader:
    # load batch of data correctly
    batches += 1
    x_batch = x_batch.view(x_batch.shape[0], -1).to(device)
    y_batch = y_batch.to(device)

    preds = torch.argmax(model(x_batch), dim=1)
    acc += (preds == y_batch).cpu().numpy().mean()

  print(f'Test accuracy {acc / batches:.3}')