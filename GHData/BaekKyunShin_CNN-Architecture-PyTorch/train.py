import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.architecture_lenet import LeNet5
from models.architecture_alexnet import AlexNet

from models.utils import accuracy, test



parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=2)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

train_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

valid_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.CIFAR10(
    root='cifar_data', 
    train=True, 
    transform=train_transforms,
    download=False)

valid_dataset = datasets.CIFAR10(
    root='cifar_data', 
    train=False,  
    transform=valid_transforms, 
    download=False)

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=args.batch_size, 
    shuffle=True)

valid_loader = DataLoader(
    dataset=valid_dataset, 
    batch_size=args.batch_size, 
    shuffle=False)


def train(data_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop

    Args:
        train_loader (DataLoader): DataLoader for training dataset
        model (class): CNN model class
        criterion (torch.nn criterion): loss function
        optimizer (torch.optim optimizer): optimizer function
        device: cuda or cpu

    Returns:
        model: same as Args' model
        optimizer: same as Args' optimizer
        epoch_loss (float): total training epoch loss
    '''
    model.train()
    running_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
    
        # Forward pass and record loss
        ouputs = model(inputs) 
        loss = criterion(ouputs, targets) 
        running_loss += loss.item() * inputs.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss / len(train_loader.dataset)

    return epoch_loss


def valid(data_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop

    Args:
        valid_loader (DataLoader): DataLoader for validating dataset
        model (class): CNN model class
        criterion (torch.nn criterion): loss function
        device: cuda or cpu

    Return:
        epoch_loss (float): total validation epoch loss
    '''
    model.eval()
    running_loss = 0
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        # Forward pass and record loss
        outputs = model(inputs) 
        loss = criterion(outputs, targets) 
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return epoch_loss


def main(print_every: int = 1):
    model = LeNet5(num_classes=10).to(device)
    # model = AlexNet(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    train_losses = []
    valid_losses = []
    # train model
    for epoch in range(args.epochs):
        train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        with torch.no_grad():
            valid_loss = valid(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if (epoch+1) % print_every == 0:
            
            train_acc = accuracy(model, train_loader, device=device)
            valid_acc = accuracy(model, valid_loader, device=device)
                
            print(f'{datetime.datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}%\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}%')
    test(valid_loader, model, device)


if __name__=="__main__":
    # get dataset and dataloader
    main()
    


    
