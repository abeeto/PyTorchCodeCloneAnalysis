from dataset import Mydatasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as Transforms
from model import Mymodel
from torch.optim import Adam
import numpy as np

gpu_avail = torch.cuda.is_available()
device = torch.device("cuda")

transforms = Transforms.Compose([Transforms.ToTensor()])
dataset_train = Mydatasets(train=True, transform=transforms)
# for i,(img,label) in enumerate(dataset_train):
#     print(i,label)
#     pass

train_loader = DataLoader(dataset_train, batch_size=5000, shuffle=True)

model = Mymodel(num_class=10)

if gpu_avail:
    model.to(device)

optimizer = Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

epochs = 100

for epoch in range(epochs):
    model.train()

    train_loss = 0.0
    train_acc = 0.0

    for batch, (img, label) in enumerate(train_loader):
        if gpu_avail:
            img = img.to(device)
            label = label.to(device)

        optimizer.zero_grad()
        outputs = model(img)
        loss = loss_fn(outputs, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*img.size(0)
        _, prediction = torch.max(outputs, 1)

        train_acc += np.sum(prediction.cpu().numpy() == label.cpu().numpy())

        print(f'epoch:{epoch}\tbatch:{batch}\tloss:{loss.item()*100}')

    train_acc = train_acc / 60000
    train_loss = train_loss / 60000
    # print(f'epoch:{epoch}\ttrain loss:{train_loss}\ttrain accuracy:{train_acc}')




