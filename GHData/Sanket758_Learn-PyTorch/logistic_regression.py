"""
Description: We will implement a logistic regression model with PyTorch
Steps: 
    0. Prepare Data
    1. Design Model, choose Loss function, choose optimizer
    2. Create a Training Loop
        a. Forward pass
        b. Calculate loss
        c. Backward pass
        d. Take optimizer step
        e. re-iterate for n_epochs

"""

import torch
import torch.nn as nn
import torch.optim

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 0: Prepare Data
# We will make use of breast cancer dataset which  is already available in sklearn
dataset = datasets.load_breast_cancer()

X, y = dataset.data, dataset.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, random_state=1337)

# Standard Scaling
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# As this data is primarily in numpy format, we need to convert it into tensors.
X_train_tensors = torch.from_numpy(X_train_scaled.astype(np.float32))
X_test_tensors = torch.from_numpy(X_test_scaled.astype(np.float32))
y_train_tensors = torch.from_numpy(y_train.astype(np.float32))
y_test_tensors = torch.from_numpy(y_test.astype(np.float32))

y_train_tensors = y_train_tensors.view(y_train_tensors.shape[0], 1)
y_test_tensors = y_test_tensors.view(y_test_tensors.shape[0], 1)

# Step 1: Model Building
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.model = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)

model = LogisticRegression(n_features, 1) # As we have only one output

# Define optimizer and loss function
criterion = nn.BCELoss() # Binary cross entropy loss becuz we are doing classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Step 3: Construct a training loop
EPOCHS = 500
for epoch in range(EPOCHS):
    # a. Forward pass
    y_hat = model(X_train_tensors)

    # b. Calculate loss
    loss = criterion(y_hat, y_train_tensors)

    # c. Backward pass
    loss.backward()

    # d. take optimizer step
    optimizer.step()

    # e. set grads to zero
    optimizer.zero_grad()

    if (epoch+1) % 50 == 0:
        print(f'Epoch {epoch+1}: Loss: {loss.item()}')


# Step 4. Evaluate
with torch.no_grad():
    y_preds = model(X_test_tensors).round()
    acc = y_preds.eq(y_test_tensors).sum() / float(y_test_tensors.shape[0])
    
print(f'Accuracy: {acc}')