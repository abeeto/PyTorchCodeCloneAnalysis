import pandas
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torch.optim as optim

# data preprocessing
data = pandas.read_csv('data/linear_800bytes_500_iterations.csv')
data = data.sort_values(by=['time_stamp'])
data = data.reset_index(drop=True)
X = data.values[:,1]
y = data.values[:,2]
for i in range(1, len(X)):
    X[i] = X[i] + X[i - 1]

# scaling X
scaler = StandardScaler()
X_norm = scaler.fit_transform(X.reshape(-1,1))
m = X_norm.shape[0]

# define model
theta = torch.tensor(torch.rand(2).view(2,1), dtype=torch.float64, requires_grad=True)
X_input = torch.cat((torch.ones([m, 1], dtype=torch.float64), torch.tensor(X_norm)), dim=1)
y_input = torch.tensor(y).view(-1,1)
n_epochs = 2000
lr = 0.01

# use Gradient Descent as optimizer, registered theta as parameter that can be optimized
optimizer = optim.SGD([theta], lr=0.01)

for epoch in range(n_epochs):
    # we set zero to clear gradient per epoch
    optimizer.zero_grad()
    
    y_pred = torch.mm(X_input, theta)
    loss = F.mse_loss(y_pred, y_input)
    
    # calculate gradient use pytorch autograd
    loss.backward()
    
    # call optimizer to update weight
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(epoch + 1, loss.item())
print(theta)

# stop autograd to track down our later calculation
theta.requires_grad_(False)

# number of bytes to predict 
test_bytes = 10

# scaling test data
X_test = torch.cat((torch.ones([1, 1], dtype=torch.float64), torch.tensor(scaler.transform([[10]]))), dim=1)

# predict
y_test = torch.mm(X_test, theta)

print(y_test.item())