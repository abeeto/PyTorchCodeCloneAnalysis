# PyTorch Refresher on MNIST Database
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

import time
from tqdm import tqdm
from easydict import EasyDict


def load_data():
    # Lines 18 - 30 taken from
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_loader = DataLoader(
        training_data,
        batch_size=128,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )

    return EasyDict(train=train_loader, test=test_loader)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(4 * 4 * 64, 200),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        )

    def forward(self, x):
        logits = self.seq(x)
        probs = F.softmax(logits, dim=1)
        return probs


def sequential_data():
    # See the comment on lines 16-17
    # Edits to reshape data to a [batch, 784, 1] dataloader for LSTM
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=Compose([
            ToTensor(),
            Lambda(lambda x: torch.reshape(torch.flatten(x), [784, 1]))
        ])
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=Compose([
            ToTensor(),
            Lambda(lambda x: torch.reshape(torch.flatten(x), [784, 1]))
        ])
    )

    train_loader = DataLoader(
        training_data,
        batch_size=200,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_data,
        batch_size=200,
        shuffle=False,
        num_workers=0
    )

    return EasyDict(train=train_loader, test=test_loader)


class LSTM(nn.Module):
    def __init__(self, n_layers=1):
        super().__init__()
        self.n_layers = n_layers
        self.lstm = nn.LSTM(1, 512, batch_first=True)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout()
        self.activation = nn.ReLU()

    def forward(self, input):
        h_0 = torch.zeros(self.n_layers, input.size(0), 512, requires_grad=True).cuda()
        c_0 = torch.zeros(self.n_layers,  input.size(0), 512, requires_grad=True).cuda()
        out, hidden = self.lstm(input, (h_0, c_0))
        out = self.dropout(hidden[0][0])
        out = self.activation(out)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        logits = self.activation(out)
        probs = F.softmax(logits, dim=1)
        return probs


def split_seconds(seconds):
    minutes = seconds // 60
    hours = minutes // 60
    days = hours // 24
    return seconds % 60, minutes % 60, hours % 24, days


def main():
    # Load data
    # data = load_data()
    data = sequential_data()

    # Define model
    # model = CNN()
    model = LSTM()

    # Cuda setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(device)

    # Optimizer setup
    optimizer = Adam(model.parameters(), lr=1e-3)
    # optimizer = SGD(
    #     model.parameters(),
    #     lr=0.01,
    #     weight_decay=1e-6,
    #     momentum=0.9,
    #     nesterov=True
    # )

    # Loss function
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    # Number of epochs
    num_epochs = 8

    # Train or load model?
    model.train()
    train_model = True
    print("Training model....")
    start = time.time()
    if train_model:
        # LSTM hidden layer
        for epoch in tqdm(range(num_epochs)):
            total_loss = 0
            for image, label in tqdm(data.train, leave=False):
                image, label = image.to(device), label.to(device)
                optimizer.zero_grad()
                # CNN
                # probabilities = model(image)
                # LSTM
                probabilities = model(image)
                loss = loss_fn(probabilities, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pass
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs} has loss {total_loss}")
        # torch.save(model.state_dict(), "mnist_cnn.pt")
        torch.save(model.state_dict(), "mnist_lstm.pt")
    else:
        # state = torch.load("mnist_cnn.pt", map_location=torch.device(device))
        state = torch.load("mnist_lstm.pt", map_location=torch.device(device))
        model.load_state_dict(state)
    end = time.time()
    seconds, minutes, hours, days = split_seconds(end - start)
    print(f"Training Runtime: {int(days)}d {int(hours)}h {int(minutes)}m {seconds}s")

    # Evaluate model on test data
    model.eval()
    print("Evaluating model....")
    start = time.time()
    num_test = 0
    num_correct = 0
    for image, label in tqdm(data.test):
        image, label = image.to(device), label.to(device)
        probabilities = model(image)
        _, pred = probabilities.max(1)
        num_test += label.size(0)
        num_correct += pred.eq(label).sum().item()
    print(f"Test accuracy: {num_correct / num_test * 100}")
    end = time.time()
    seconds, minutes, hours, days = split_seconds(end - start)
    print(f"Training Runtime: {int(days)}d {int(hours)}h {int(minutes)}m {seconds}s")


if __name__ == "__main__":
    main()
