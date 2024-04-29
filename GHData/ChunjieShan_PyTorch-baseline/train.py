#!/usr/bin/python3
# -*- coding: utf8 -*-

import torch
import torch.optim as optim
import os
from torchvision import transforms, datasets
from tensorboardX import SummaryWriter
from net import SimpleConv3Net
from torch.autograd import Variable

writer = SummaryWriter()

data_dir = './data/'

data_transforms = {
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(48),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val':
    transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}

image_dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x],
                                   batch_size=16,
                                   shuffle=True,
                                   num_workers=4)
    for x in ['train', 'val']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


def train_model(model, criterion, optimizer, scheduler, num_epochs, use_gpu):
    for epoch in range(num_epochs):
        print("Epoch: {} / {}".format(epoch, num_epochs))
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_acc = 0.0
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            for data in image_dataloaders[phase]:
                inputs, labels = data

                if use_gpu == True:
                    model.cuda()
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())

                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)
                # print("Prediction: ", preds)

                if phase == "train":
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                running_loss += loss.data.item()
                running_acc += torch.sum(preds == labels).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc / dataset_sizes[phase]

            print("{}, Loss: {:.4f}, Acc: {:.4%}".format(
                phase, epoch_loss, epoch_acc))

            if phase == "train":
                writer.add_scalar("data/trainloss", epoch_loss, epoch)
                writer.add_scalar("data/trainacc", epoch_acc, epoch)

            if phase == "val":
                writer.add_scalar("data/valloss", epoch_loss, epoch)
                writer.add_scalar("data/valacc", epoch_acc, epoch)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

    return model


if __name__ == "__main__":
    model = SimpleConv3Net()
    print(model)
    num_epochs = 300
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    use_gpu = torch.cuda.is_available()

    my_model = train_model(model, criterion, optimizer, scheduler, num_epochs,
                           use_gpu)
