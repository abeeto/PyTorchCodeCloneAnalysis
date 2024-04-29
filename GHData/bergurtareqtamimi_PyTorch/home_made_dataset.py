import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# home made data set where second bit determines outcome(small train, large test)
x = [[1,1,0,1,0],[1,0,0,1,0],[0,1,0,0,0],[1,1,1,1,0],[1,0,0,0,0],[0,0,0,1,0],[1,1,0,1,1]]
y = [[1],[0],[1],[1],[0],[0],[1]]
X = [[1,1,1,1,1],[1,0,1,1,1],[1,1,0,0,1], [0,0,0,0,1], [0,0,0,0,0]]
Y = [[1],[0],[1],[0],[0]]

# convert to tensors
x = torch.as_tensor(x, dtype=torch.float32)
y = torch.as_tensor(y, dtype=torch.float32)
X = torch.as_tensor(X, dtype=torch.float32)
Y = torch.as_tensor(Y, dtype=torch.float32)

# wrapping label and input together in one tensor
training_data = TensorDataset(x, y)
train_loader = DataLoader(training_data , batch_size = 1, shuffle=True)
test_data = TensorDataset(X, Y)
test_loader = DataLoader(test_data , batch_size = 1, shuffle=False)

# create basic neural network
class NN(nn.Module):
    def __init__(self) -> None:
        super(NN, self).__init__()
        self.l1 = nn.Linear(5, 5, dtype=torch.float32) # define input type
        self.out = nn.Linear(5, 1, dtype=torch.float32)
        self.a = nn.Sigmoid() # perfect activation when predicted outcome is between 0 and 1
    
    def forward(self, x):
        x = self.l1(x)
        x = self.out(x)
        x = self.a(x)
        return x

# used for tesing since the only two outcomes are 0 and 1
def translator(x: float) -> float:
    if x >= 0.5:
        return 1.0
    return 0.0

model = NN()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.L1Loss() # Mean absolute error -> | a - b |

# training
for epoch in range(3000):
    for data in train_loader:
        x, y = data

        # zero the parameter gradients
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(y, output)
        loss.backward()
        optimizer.step()        

# testing
with torch.no_grad():
    N = 0
    correct = 0
    for data in test_loader:
        x, y = data
        output = model(x)

        print(f'pred: {output.item()} actual: {y}')
        N += 1
        if translator(output.item()) == y.item():
           correct += 1

print(f'Accuracy: {correct / N}') 




