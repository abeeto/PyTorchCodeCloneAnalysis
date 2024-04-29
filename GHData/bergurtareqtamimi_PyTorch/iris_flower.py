import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

name_mapper = {'Iris-setosa': [1, 0, 0], 'Iris-versicolor': [0, 1, 0], 'Iris-virginica': [0, 0, 1]}

data = pd.read_csv('data/IRIS.csv')

x = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values

x, X, y, Y = train_test_split(x, y, test_size=0.2, random_state=42)

y = [name_mapper[y[i]] for i in range(len(y))]
Y = [name_mapper[Y[i]] for i in range(len(Y))]


x = torch.from_numpy(x)
X = torch.from_numpy(X)
y = torch.as_tensor(y, dtype=torch.float32)
Y = torch.as_tensor(Y, dtype=torch.float32)

training_data = TensorDataset(x, y)
train_loader = DataLoader(training_data , batch_size = 1, shuffle=True)
test_data = TensorDataset(X, Y)
test_loader = DataLoader(test_data , batch_size = 1, shuffle=False)


class NN(nn.Module):
    def __init__(self) -> None:
        super(NN, self).__init__()
        self.l1 = nn.Linear(4, 7, dtype=torch.double)
        self.out = nn.Linear(7, 3, dtype=torch.double)
        self.a = nn.Softmax()
    
    def forward(self, x):
        x = self.l1(x)
        x = self.out(x)
        x = self.a(x)
        return x


model = NN()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss() # good for classification problems

# training
for epoch in range(1000):
    for data in train_loader:
        x, y = data

        optimizer.zero_grad()

        # calculate loss, backprobagation and optimize
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

        print(f'pred: {output.tolist()} actual: {y.tolist()}')
        N += 1
        if output.tolist()[0].index(max(output.tolist()[0])) == y.tolist()[0].index(max(y.tolist()[0])):
           correct += 1

print(f'Accuracy: {correct / N}')