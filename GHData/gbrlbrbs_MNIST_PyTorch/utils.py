import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import pandas as pd
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from datetime import datetime
from constants import *
from lenet5 import LeNet5

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_accuracy(net, data_loader):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        net.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            _, y_prob = net(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n


def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''
    
    # temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    fig.show()

    plt.savefig('losses.png', format=FIG_FORMAT)
    
    # change the plot style to default
    plt.style.use('default')


def train(train_loader, net, criterion, optimizer, print_every=1000):
    net.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):

        optimizer.zero_grad()

        inp = data[0].to(device)
        target = data[1].to(device)

        output, _ = net(inp)
        loss = criterion(output, target)
        running_loss += loss.item() * inp.size(0)

        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)

    return net, optimizer, epoch_loss


def validate(valid_loader, net, criterion):

    net.eval()
    running_loss = 0.0

    for i, data in enumerate(valid_loader, 0):

        inp = data[0].to(device)
        target = data[1].to(device)

        output, _ = net(inp)
        loss = criterion(output, target)

        running_loss += loss.item() * inp.size(0)
    
    epoch_loss = running_loss / len(valid_loader.dataset)

    return net, epoch_loss


def training_loop(net, criterion, optimizer, train_loader, valid_loader, print_every=1):
    
    best_loss = math.inf
    train_losses = []
    valid_losses = []

    for epoch in range(NUM_EPOCHS):
        
        print(f'{datetime.now().time().replace(microsecond=0)} --- '
              f'Epoch: {epoch+1}\n')

        # training
        print('Training...')
        net, optimizer, train_loss = train(train_loader, net, criterion, optimizer)
        train_losses.append(train_loss)
        train_accuracy = get_accuracy(net, train_loader)
        print(f'Train loss: {train_loss:.4f}')
        print(f'Train accuracy: {100 * train_accuracy:.2f}\n')

        # validation
        print('Validating...')
        with torch.no_grad():
            net, valid_loss = validate(valid_loader, net, criterion)
            valid_losses.append(valid_loss)
        valid_accuracy = get_accuracy(net, valid_loader)
        print(f'Validation loss: {valid_loss:.4f}')
        print(f'Validation accuracy: {100 * valid_accuracy:.2f}\n')

    torch.save(net.state_dict(), SAVE_PATH)
            
    plot_losses(train_losses, valid_losses)

    return net, optimizer, (train_losses, valid_losses)


def get_data():
    # download and create datasets
    train_dataset = datasets.MNIST(
        root='data', train=True, transform=transform, download=True
    )

    valid_dataset = datasets.MNIST(
        root='data', train=False, transform=transform
    )

    # define the data loaders
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    return train_loader, valid_loader, train_dataset, valid_dataset


def get_whole_model():
    net = LeNet5(NUM_CLASSES).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    return net, criterion, optimizer


def evaluate_pictures(dataset, net=None):
    if net is None:
        net = LeNet5(NUM_CLASSES).to(device)
        net.load_state_dict(torch.load(SAVE_PATH))

    figure = plt.figure(figsize=FIG_SIZE, dpi=400)
    net.eval()

    for idx in range(1, NUM_ROWS*NUM_COLUMNS + 1):
        plt.subplot(NUM_ROWS, NUM_COLUMNS, idx)
        plt.axis('off')
        plt.imshow(dataset.data[idx], cmap='gray_r')

        with torch.no_grad():
            inp = dataset[idx][0].unsqueeze(0)
            inp = inp.to(device)
            _, probs = net(inp)

        title = f'{torch.argmax(probs)} ({torch.max(probs * 100):.0f}%)'
        plt.title(title, fontsize=7)

    figure.suptitle('LeNet 5 - Predictions')
    plt.savefig('predictions.png', format=FIG_FORMAT)


def plot_confusion_matrix(valid_loader: DataLoader, net=None):
    if net is None:
        net = LeNet5(NUM_CLASSES).to(device)
        net.load_state_dict(torch.load(SAVE_PATH))

    figure = plt.figure(figsize=FIG_SIZE, dpi=400)
    net.eval()

    y_pred = []
    y_true = []

    for inputs, labels in valid_loader:
        output, _ = net(inputs.to(device))
        output = torch.max(torch.exp(output), 1)[1].data.cpu().numpy()
        y_pred.extend(output)

        lbls = labels.data.cpu().numpy()
        y_true.extend(lbls)

    classes = [i for i in range(10)]
    cf = confusion_matrix(y_true, y_pred, labels=classes)

    sns.heatmap(cf, annot=True)
    plt.savefig('confusion_matrix.png')