import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torch.utils import data
from torchvision.datasets import cifar
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.autograd.variable import Variable
from torch.utils.data.sampler import SubsetRandomSampler

print(f'is CUDA available? {torch.cuda.is_available()}')
tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

root = Path('/data/dataset/amazon')
train_dir = root / 'train'
test_dir = root / 'test'
bs = 20
train_df = pd.read_csv(root / 'train_v2.csv')
labels = pd.Series(train_df['tags'].values, index=train_df['image_name'])
labels = {k: v.split() for k, v in labels.iteritems()}
tags = list(labels.values())
tags = tuple(set(item for sublist in tags for item in sublist))
tags2index = {tag: i for i, tag in enumerate(tags)}
print(f'tag lens = {len(tags)}')


def encode_tags(input_tags):
    out = np.zeros((1, len(tags)))
    for t in input_tags:
        out[:, tags2index[t]] = 1
    out = out.squeeze(0)
    out = torch.from_numpy(out)
    return out


fns = list(labels.keys())
print(f'len of dataset = {len(fns)}')
idxs = list(range(len(fns)))
idxs2fn = {i: fn for i, fn in zip(idxs, fns)}
validation_split = .1
random_seed = 42
split = int(np.floor(validation_split * len(fns)))
np.random.seed(random_seed)
np.random.shuffle(idxs)
train_ids, val_ids = idxs[split:], idxs[:split]
train_sampler = SubsetRandomSampler(train_ids)
val_sampler = SubsetRandomSampler(val_ids)


class AmazonDataset(data.Dataset):
    def __init__(self, transform):
        self.transform = transform

    def __len__(self):
        return len(labels)

    def __getitem__(self, index):
        fn = fns[index]
        fn_tags = labels[fn]
        fn_tags = encode_tags(fn_tags)
        img = Image.open(train_dir / f'{fn}.jpg')

        if self.transform:
            img = self.transform(img)
        img = img[:3, ...]
        return img, fn_tags


trainset = AmazonDataset(transform=tfms)
train_loader = data.DataLoader(trainset, batch_size=bs, sampler=train_sampler)
val_loader = data.DataLoader(trainset, batch_size=bs, sampler=val_sampler)
model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.Linear(1024, 512),
    nn.Linear(512, 256),
    nn.Linear(256, len(tags) * 2),
)
model.cuda()
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))
crit = nn.BCEWithLogitsLoss()
epocs = 10
print_debug = 10
validation_run = 500
save_run = 100


def train(epocs):
    for e in range(epocs):
        train_losses = []
        for i, (x, y) in enumerate(train_loader):
            model.train(True)
            x, y = x.cuda().float(), y.cuda().float()
            yh = model(x)
            yh = yh.reshape(x.shape[0], -1, 2)
            yh = F.softmax(yh, dim=2)
            _, yh = torch.max(yh, dim=2)
            yh = yh.float()
            loss = crit(y, yh)
            loss = Variable(loss, requires_grad=True)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % print_debug == 0:
                print(f'[{e:5d} {i:5d}/{len(train_loader)}] train = {np.mean(train_losses):.5f}')

            if i > 0 and i % validation_run == 0:
                torch.save(model.state_dict(), f'checkpoints/{e}_{i}.pt')
                model.train(False)
                val_losses = []
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val, y_val = x_val.cuda().float(), y_val.cuda().float()
                        yh = model(x_val)
                        yh = yh.reshape(x_val.shape[0], -1, 2)
                        yh = F.softmax(yh, dim=2)
                        _, yh = torch.max(yh, dim=2)
                        yh = yh.float()
                        loss = crit(y_val, yh)
                        val_losses.append(loss.item())

                    print(f'[{e:5d}] val = {np.mean(val_losses):.5f}')


model_size = sum([1 for i in model.children()])
print(f'model size = {model_size}')
for i, child in enumerate(model.children()):
    if i < model_size - 1:
        child.requires_grad = False

train(2)
