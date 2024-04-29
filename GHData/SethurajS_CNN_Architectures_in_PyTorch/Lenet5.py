# Importing the requirements
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Lenet Model

class Lenet5(nn.Module):
    def __init__(self, in_channels=1, out_classes=10):
        super(Lenet5, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(120*1*1, 84),
            nn.Linear(84, out_classes)
        )

    def forward(self, x):
        x = self.network(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


if __name__ == "__main__":

    # Device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Defining Hyperparameters
    batch_size = 64
    num_epochs = 1
    learning_rate = 0.001

    # Initialize the CNN model
    model = Lenet5().to(device=device)

    # Download and load datasets
    train_data = datasets.MNIST(root="data/", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data = datasets.MNIST(root="data/", train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # Optmizer and Loss Function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\n")
    print(">>>> Training !!")
    print("\n")
    # Training the Lenet model
    for epoch in range(num_epochs):
        for data_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()


    # Evaluating the model
    def evaluate(dataset, model):
        if dataset.dataset.train:
            print(">>>> Evaluating accuracy on train dataset :")
        else:
            print("\n")
            print(">>>> Evaluating accurcy on test dataset : ")
        num_correct = 0
        num_samples = 0
        model.eval()
        with torch.no_grad():
            for X, y in dataset:
                # Get data to cuda
                X = X.to(device=device)
                y = y.to(device=device)

                # Prediction
                predictions = model(X)
                _, prediction = predictions.max(1)
                num_correct += (prediction == y).sum()
                num_samples += prediction.size(0)

            print("     Got {}/{} with accuracy of {}% !".format(num_correct, num_samples, (float(num_correct)/float(num_samples))))

        model.train()

    # Evaluating 
    evaluate(dataset=train_loader, model=model)
    evaluate(dataset=test_loader, model=model)
