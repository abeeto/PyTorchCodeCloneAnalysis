import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/mnist2")

device = "cuda" if torch.cuda.is_available else "cpu"

# seed
np.random.seed(777)
torch.manual_seed(777)

# GPU seed
if device == "cuda":
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epoch = 10
batch_size = 100

mnist_train = datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)


data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First Layer
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv -> (?, 28, 28, 32)
        #    Pool -> (?, 14, 14, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Second Layer
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv -> (?, 14, 14, 64)
        #    Pool -> (?, 7, 7, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 전결합층 7*7*64 -> 10 outputs
        self.fc = nn.Linear(7*7*64, 10,bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = nn.Flatten()(out)
        out = self.fc(out)
        return out


model = CNN().to(device)


criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
print(f"Total Batch : {total_batch}")

running_loss = 0.0
running_correct = 0

for epoch in range(training_epoch):

    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)
        
        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        
        running_loss = cost / total_batch

        predicted = torch.argmax(hypothesis, 1) == Y
        running_correct = predicted.float().mean()

    writer.add_scalar("training_loss", running_loss, (epoch+1)*batch_size)    
    writer.add_scalar("accuracy", running_correct*100, (epoch+1)*batch_size)
    print(f'[Epoch {epoch+1:>4} cost = {running_correct*100:.4f}]')
    running_loss = 0.0
    running_correct = 0

with torch.no_grad():
    X_test = mnist_test.data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.targets.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print(f"Accuracy: {accuracy.item()}")

    