import torch
import torchvision
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # All functions that don't have any parameters
from torch.utils.data import DataLoader # Gives easier dataset management and creates mini batches
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms # Transformations we can perform on our dataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 28
sequence_length = 28 # Assume one row at a time at each timestep
# MNIST dataset -> 28 sequences which each has 28 features (이미지를 row by row로 인풋으로 쓰는 것임)
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# Create a bidirectional LSTM
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True) # MNIST dataset에는 batch 크기가 맨 앞 디멘션임
        self.fc = nn.Linear(hidden_size, num_classes) # fully connected layer

    def forward(self, x): # hidden state, cell state를 lstm에 보낼 것
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # num_layers개 * 2인 이유는 하나는 forward, 하나는 backward로 갈 것이기 때문. -> 그리고 concatenated 된 것이 해당 time sequence
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # num_layers개 * 2인 이유는 하나는 forward, 하나는 backward로 갈 것이기 때문. -> 그리고 concatenated 된 것이 해당 time sequence
        
        out, _ = self.lstm(x, (h0, c0)) # cell state는 활용하지 않음
        out = self.fc(out[:, -1, :]) # Last hidden state를 활용한다

        return out

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
# MNIST DATASET: Nx1x28x28 형태를 갖고 있음
# RNN은 Nx28x28 형태를 원함


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize the network
model = BiLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device).squeeze(1)
        targets = targets.to(device)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()

# Check accuracy on training & test to see how good is our model
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
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)
            
            scores = model(x)
            # shape 64 X 10
            _, predictions = scores.max(1) # give us the value. 우리는 index of maximum value를 알고 싶다
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
        
