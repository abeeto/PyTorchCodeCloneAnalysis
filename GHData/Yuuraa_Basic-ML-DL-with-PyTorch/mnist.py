# Imports
import torch
import torch.nn as nn
# Stochastic gradient descent나 Adam같은 optimizer들
import torch.optim as optim
# ReLU 같은 활성화 함수들
import torch.nn.functional as F
from torch.utils.data import DataLoader
# 여기에서 MNIST 데이터를 받아올 것이다
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        # Parent class의 initializer를 불러옴. 여기서는 nn.Module
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Dummy data
model = NN(784, 10)
# 64는 동시에 돌릴 데이터의 수 즉, mini batch size
x = torch.randn(64, 784)
print(model(x).shape) # torch.Size([64, 10])


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load Data
# dataset이라는 폴더를 만들고 저장함
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True) # 데이터셋이 없다면 다운로드 한다
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # Shuffle the batches. 매 epoch마다 다를 수 있다
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True) # 데이터셋이 없다면 다운로드 한다
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True) # Shuffle the batches. 매 epoch마다 다를 수 있다

# Initialize the network
model = NN(input_size, num_classes=num_classes).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get tensor data to cuda if possible
        data = data.to(device = device)
        targets = targets.to(device = device)

        #print(data.shape) #torch.Size(64, 1, 28, 28) 1은 흑백 사진이기 때문으로, RGB라면 3이다. 28, 28은 이미지의 w와 h
        # Get to correct shape
        data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad() # Set all the gradients to zero for each batch
        loss.backward()

        # gradient descent or adam step
        optimizer.step()


# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)            
            
            scores = model(x)
            # shape 64 X 10
            _, predictions = scores.max(1) # give us the value. 우리는 index of maximum value를 알고 싶다
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()
    #return acc

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
