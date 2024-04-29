import torch
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
import copy


DATA_DIR = 'C:\\Users\\45569\\Pictures\\Camera Roll\\'
MODEL_NAME = 'resnet'
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCH = 5
FEATURE_EXTRACT = True
INPUT_SIZE = 224

all_imgs = datasets.ImageFolder(DATA_DIR + 'train', transform=transforms.Compose([
    # transforms.Resize(INPUT_SIZE),
    transforms.RandomResizedCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

# print(all_imgs[0][0].shape)
# transforms.ToPILImage()(all_imgs[0][0]).show()
loader = Data.DataLoader(all_imgs, batch_size=BATCH_SIZE, shuffle=True)


def initialize_model(model_name=MODEL_NAME, feature_extract=FEATURE_EXTRACT, pretrained=False):
    if model_name == 'resnet':
        model = models.resnet50(pretrained=pretrained)

        if not pretrained:
            pre = torch.load(r'D:\BaiduNetdiskDownload\models\resnet50-19c8e357.pth')
            model.load_state_dict(pre)

        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

        return model


model = initialize_model()
print(model)

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "test": transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

image_datasets = {x: datasets.ImageFolder(DATA_DIR+x, transform=data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: Data.DataLoader(image_datasets[x],
                                  batch_size=BATCH_SIZE, shuffle=True) for x in ['train', 'test']}

# transforms.ToPILImage()(image_datasets['test'][20][0]).show()


def train_model(model, dataloaders, loss_fn, optimizer, epochs=EPOCH):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 0.
    for epoch in range(EPOCH):
        for phase in ['train', 'test']:
            running_loss = 0.
            running_correct = 0.
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for inputs, labels in dataloaders[phase]:
                with torch.autograd.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)

                preds = outputs.argmax(dim=1)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds.view(-1) == labels.view(-1)).item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_accuracy = running_correct / len(dataloaders[phase].dataset)

            print('Phase {} loss: {}, accuracy: {}'.format(phase, epoch_loss, epoch_accuracy))

            if phase == 'test' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
# train_model(model, dataloaders, loss_fn, optimizer, EPOCH)