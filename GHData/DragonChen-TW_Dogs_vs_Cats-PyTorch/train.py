import torch
from torch import nn
from torch.utils.data import DataLoader
import json
#
import models
from data.datasets import DogCatData
from val import val


def train(model, data, epoch, criterion, optimizer, device='cuda'):
    print('==========Epoch {}=========='.format(epoch))
    model.train()

    for i, (image, label) in enumerate(data):
        # == Pre-Process ==
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        score = model(image)
        loss = criterion(score, label)
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            print('epoch {} ({} / {}) loss: {}'.format(
                epoch, i, len(data),
                loss.item()
            ))

if __name__ == '__main__':
    # == General ==
    # device = torch.device('cpu')
    device = torch.device('cuda')

    # == Model ==
    model = models.LeNet()
    model = model.to(device)

    # == Load Data ==
    root = 'D:/data/dogs_cats'
    dataset_train = DogCatData(root, mode='train')
    dataset_val  = DogCatData(root, mode='val')
    train_data = DataLoader(dataset_train,
        shuffle=True, batch_size=32, num_workers=4)
    val_data = DataLoader(dataset_val,
        shuffle=True, batch_size=32, num_workers=4)

    # == optimizer ==
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # == Main Loop ==
    max_acc = 0
    max_epoch = 1
    for epoch in range(max_epoch):
        train(model, train_data, epoch, criterion, optimizer)
        acc = val(model, val_data)
        if acc > max_acc:
            max_acc = acc
            torch.save(model, 'checkpoints/lenet_max.pt')

    print('==========Max Acc: {}=========='.format(max_acc))
