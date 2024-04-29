# %%
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as T
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
import model
from tqdm import tqdm
import random
import time
import torchsummary
NUM_WORKERS = 8
BATCH_SIZE = 64
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = T.Compose([
    T.Resize((220, 220)),
    T.RandomHorizontalFlip(),
    T.ToTensor()
])

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %%
train_dataset = CIFAR10(root='./data', download=True, transform=train_transform, train=True)
# train_dataset.data = train_dataset.data[:10_000]
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


# %%
def train_step(batch, model, optimizer, loss_fn):
    inputs, targets = batch[0].to(DEVICE), batch[1].to(DEVICE)
    optimizer.zero_grad()
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()
    return loss


def predict_step(batch, model, loss_fn):
    inputs, targets = batch[0].to(DEVICE), batch[1].to(DEVICE)
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    return logits, loss


# %% =================================================================================================
base_model = model.BaseModel(num_classes=10)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-4)
base_loss = []
base_model.to(DEVICE)
print(f'Parameters: {count_parameters(base_model):,}')
start = time.time()
base_step_loss = []
for epoch in range(EPOCHS):
    print(f'Epoch: {epoch+1}')
    t_loss = []

    for batch in tqdm(train_loader):
        loss = train_step(batch, base_model, optimizer, loss_fn)
        batch_loss = loss.detach().cpu().numpy().item()
        t_loss.append(batch_loss)
        base_step_loss.append(batch_loss)

    print(f'train loss: {np.mean(t_loss):.4f}')
    base_loss.append(np.mean(t_loss))
base_time = time.time() - start
print(f'Time to train: {base_time}')
torch.cuda.empty_cache()
print()


# %% =================================================================================================
atrous_model = model.AtrousModel(num_classes=10)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(atrous_model.parameters(), lr=1e-4)
atrous_loss = []
atrous_step_loss = []
atrous_model.to(DEVICE)
print(f'Parameters: {count_parameters(atrous_model):,}')
start = time.time()
for epoch in range(EPOCHS):
    print(f'Epoch: {epoch+1}')
    t_loss = []

    for batch in tqdm(train_loader):
        loss = train_step(batch, atrous_model, optimizer, loss_fn)
        batch_loss = loss.detach().cpu().numpy().item()
        t_loss.append(batch_loss)
        atrous_step_loss.append(batch_loss)

    print(f'train loss: {np.mean(t_loss):.4f}')
    atrous_loss.append(np.mean(t_loss))

atrous_time = time.time() - start
print(f'Time to train: {atrous_time}')
torch.cuda.empty_cache()
print()


# %% =================================================================================================
depth_model = model.DepthModel(num_classes=10)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(depth_model.parameters(), lr=1e-4)
depth_loss = []
depth_step_loss = []
depth_model.to(DEVICE)
print(f'Parameters: {count_parameters(depth_model):,}')
start = time.time()
for epoch in range(EPOCHS):
    print(f'Epoch: {epoch+1}')
    t_loss = []

    for batch in tqdm(train_loader):
        loss = train_step(batch, depth_model, optimizer, loss_fn)
        batch_loss = loss.detach().cpu().numpy().item()
        t_loss.append(batch_loss)
        depth_step_loss.append(batch_loss)

    print(f'train loss: {np.mean(t_loss):.4f}')
    depth_loss.append(np.mean(t_loss))

depth_time = time.time() - start
print(f'Time to train: {depth_time}')
torch.cuda.empty_cache()
print()


# %%
x = np.arange(len(base_step_loss))
plt.title('CNN operations Comparison')
plt.plot(x[::20], base_step_loss[::20], label=f'7x7\n{base_time:.3f} sec')
plt.plot(x[::20], atrous_step_loss[::20], label=f'Atrous\n{atrous_time:.3f} sec')
plt.plot(x[::20], depth_step_loss[::20], label=f'Depth(Torch)\n{depth_time:.3f} sec')
plt.xlabel('epoch')
plt.ylabel('train loss')
plt.legend()
plt.savefig('loss.jpg')


# %% ================== INFERENCE TIME =======================
def run_inference(model):
    model.eval()
    with torch.no_grad():
        start = time.time()
        for batch in train_loader:
            _, _ = predict_step(batch, model, loss_fn)

    end_time = time.time() - start
    return end_time


print(f'Base model..: {run_inference(base_model)}')
print(f'Atrous model: {run_inference(atrous_model)}')
print(f'Depth model.: {run_inference(depth_model)}')
