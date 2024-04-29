import torch
from config import *
from engine import *
from model import MNISTModel
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import random_split

data = MNIST(root='data/',
                        download=True,
                        transform=ToTensor())

train, validation = random_split(dataset=data, lengths=[TRAIN_SAMPLES, VALID_SAMPLES])
train_dataloader = DataLoader(train, batch_size=TRAIN_BATCH_SIZE)
validation_dataloader = DataLoader(validation, batch_size=VALID_BATCH_SIZE)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = MNISTModel()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    train_fn(train_dataloader, model, TRAIN_BATCH_SIZE, optimizer, device)
    model.eval()
    valid_fn(validation_dataloader, model, VALID_BATCH_SIZE, device)

torch.save(model.state_dict(), MODEL_NAME)
print('model saved.')








