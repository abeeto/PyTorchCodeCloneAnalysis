from pyexpat import model
import torch, os, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from dataloader import get_loader
from dataset import get_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset, val_dataset, test_dataset = get_dataset(IMG_SIZE=224)
train_dataloader, val_dataloader, test_dataloader = get_loader(BATCH_SIZE=64)

# model_ft = models.resnet50(pretrained=True)
model_ft = models.mobilenet_v2(pretrained=True)

for param in model_ft.parameters():
    param.requires_grad = False

# num_fc_ftr = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_fc_ftr, len(train_dataset.classes))

model_ft.classifier[1] = nn.Linear(1280, len(train_dataset.classes))

model_ft = model_ft.to(DEVICE)
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam([{'params':model_ft.fc.parameters()}], lr=0.001)
optimizer = torch.optim.Adam([{'params':model_ft.classifier[1].parameters()}], lr=0.001)


def train(model, device, train_loader, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        x, y = data
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
    print ('Train Epoch: {}\t Loss: {:.6f}'.format(epoch,loss.item()))

def val(model, device, loader, dataset):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(loader):          
            x, y= data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            test_loss += criterion(y_hat, y).item() # sum up batch loss
            pred = y_hat.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(loader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(dataset),
        100. * correct / len(dataset)))

for epoch in range(1, 10):
    print('Epoch: ', epoch)
    train(model=model_ft,device=DEVICE, train_loader=train_dataloader, epoch=epoch)
    val(model=model_ft, device=DEVICE, loader=val_dataloader, dataset=val_dataset)

torch.save(model_ft, './models/model_mobilenetv2.pt')