import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import optuna
from torchvision import models


train_path = "C:\\Users\\Pupil\\Desktop\\kek\\data\\train"
val_path = "C:\\Users\\Pupil\\Desktop\\kek\\data\\val"
image_size = (224, 224)
batch_size = 64


def train(model, epoch, train_ds):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    total_num = len(train_ds.dataset)
    train_loss = 0
    correct_num = 0
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0003)
    loss_fun = F.cross_entropy


    for image, label in train_ds:
        image = image.to(device)
        label = label.to(device)
        # Convert the tag from int32 type to long type, otherwise the calculation loss will report an error
        label = label.to(torch.long)

        output = model(image)
        loss = loss_fun(output, label)
        train_loss += loss.item() * label.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predict = torch.argmax(output, dim=-1)
        correct_num += label.eq(predict).sum()

    train_loss = train_loss / total_num
    train_acc = correct_num / total_num
    print('epoch: {} --> train_loss: {:.6f} - train_acc: {:.6f} - '.format(
        epoch, train_loss, train_acc), end='')


def evaluate(model, eval_ds, mode='val'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    loss_fun = F.cross_entropy


    total_num = len(eval_ds.dataset)
    eval_loss = 0
    correct_num = 0

    for image, label in eval_ds:
        image = image.to(device)
        label = label.to(device)
        label = label.to(torch.long)

        output = model(image)
        loss = loss_fun(output, label)
        eval_loss += loss.item() * label.size(0)

        predict = torch.argmax(output, dim=-1)
        correct_num += label.eq(predict).sum()

    eval_loss = eval_loss / total_num
    eval_acc = correct_num / total_num

    print('{}_loss: {:.6f} - {}_acc: {:.6f}'.format(
        mode, eval_loss, mode, eval_acc))
    return eval_acc

        

def objective(trial):
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(0.4),
        transforms.RandomGrayscale(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root=train_path, transform=transform)
    val_dataset = ImageFolder(root=val_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    backbone_name = trial.suggest_categorical("backbone", ["resnet18", "resnet34", "resnet50"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    model = getattr(models, backbone_name)(pretrained=True)
   
    for name, m in model.named_parameters():
        if name.split('.')[0] != 'fc':
            m.requires_grad_(False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0003)
    loss_fun = F.cross_entropy

    for epoch in range(2):
        train(model, epoch, train_loader)
        acc = evaluate(model, val_loader)
    
    return acc
  
       
    
if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=5)