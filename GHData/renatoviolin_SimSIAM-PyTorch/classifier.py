# %%
try:
    get_ipython()
    NUM_WORKERS = 0
except:
    NUM_WORKERS = 8

from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from models import ClassifierModel
import dataset
from tqdm import tqdm
import glob
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = 50


# %%
train_data = torchvision.datasets.ImageFolder('../data_cat_dog/label/train')
train_dataset = dataset.ClassifierDataset(train_data)
train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)


# %%
def acc_fn(logits, targets):
    logits = torch.argmax(logits, dim=1).detach().cpu()
    targets = targets.detach().cpu()
    acc = torch.mean((logits == targets) * 1.0)
    return acc.numpy()


def train_batch(batch, model, optimizer):
    img, label = batch[0].to(DEVICE), batch[1].to(DEVICE)
    optimizer.zero_grad()
    logits = model(img)
    loss = loss_fn(logits, label)
    loss.backward()
    optimizer.step()
    return logits, loss.item(), acc_fn(logits, label)


# %% ================== CLASSIFIER FROM SCRATCH ======================
net = ClassifierModel().to(DEVICE)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

scratch_loss = []
for epoch in range(EPOCHS):
    scratch_t_loss = []
    scratch_t_acc = []
    print(f'Epoch {epoch+1}')
    for batch in tqdm(train_dataloader):
        _, loss, acc = train_batch(batch, net, optimizer)
        scratch_t_loss.append(loss)
        scratch_t_acc.append(acc)
        scratch_loss.append(loss)

    print(f'loss: {np.mean(scratch_t_loss):.4f}  -  acc: {np.mean(scratch_t_acc):.4f}')


# %% ================== CLASSIFIER FROM SIAM PRE-TRAINED ======================
net = ClassifierModel(checkpoint='siam.pt').to(DEVICE)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

sim_loss = []
for epoch in range(EPOCHS):
    sim_t_loss = []
    sim_t_acc = []
    print(f'Epoch {epoch+1}')
    for batch in tqdm(train_dataloader):
        _, loss, acc = train_batch(batch, net, optimizer)
        sim_t_loss.append(loss)
        sim_t_acc.append(acc)
        sim_loss.append(loss)

    print(f'train loss: {np.mean(sim_t_loss):.4f}  -  train acc: {np.mean(sim_t_acc):.4f}')


# %% ======== PLOT RESULTS =======================
x = np.arange(len(scratch_loss))
plt.title('Frozen Backbone - Classifier loss')
plt.plot(x[::20], scratch_loss[::20], label='Random weights')
plt.plot(x[::20], sim_loss[::20], label='SimSIAM weights')
plt.legend()
plt.savefig('loss.jpg')
