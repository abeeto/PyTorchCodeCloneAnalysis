import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image

import os

from pnas import PNASNet5Large

device = 'cuda'

dataset_path = os.path.dirname(os.path.realpath(__file__)) + '/dataset'


def mytransform(img):
    width, height = img.size
    pad_to = max(width, height)
    transformed = torchvision.transforms.functional.pad(img, (pad_to - width, pad_to - height))
    transformed = torchvision.transforms.functional.resize(transformed, (331, 331))
    return transformed


transform = transforms.Compose([
    mytransform,
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # transforms.RandomErasing()
])

# Makes results deterministic
torch.manual_seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Loads and scripts the model
model = PNASNet5Large(num_classes=5).to(device)
model = torch.jit.script(model)

train_img_folder = torchvision.datasets.ImageFolder(
    dataset_path,
    transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_img_folder,
    batch_size=4,
    shuffle=True,
    num_workers=1,
    pin_memory=True)

lr = 1e-5

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr)

m = nn.LogSoftmax(dim=1).to(device)
loss_fn = nn.NLLLoss().to(device)


def evaluate(model, loader, loss_fn):
    loss = 0
    num_correct = 0
    with torch.no_grad():
        model.eval()
        for idx, (input, target) in enumerate(loader):
            out = model(input.to(device))
            target = target.to(device)
            loss += loss_fn(m(out), target).item()
            predictions = torch.argmax(out, dim=1)
            num_correct += torch.sum(target == predictions)
        return loss, num_correct.item()


model.train()
for epoch in range(0, 2):
    for idx, (input, target) in enumerate(train_loader):
        out = model(input.to(device))
        loss = loss_fn(m(out), target.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for idx, (input, target) in enumerate(train_loader):
    sample_out = model(input.cuda())
    break

# Validates the model produces the same outputs consistently
fixture = torch.tensor([
    [-0.6241,  1.2502,  1.1059, -0.6813, 0.8837],
    [1.3981,  0.7711, -0.3744,  0.6056, -0.7757],
    [0.2892, -0.0913, -0.3163,  0.9425, -0.7759],
    [-0.1292,  1.3259, -0.1468,  0.0087,  0.3588]], device=device)

assert torch.allclose(fixture, sample_out, rtol=0, atol=1e-4)
