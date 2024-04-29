#  Logistic_Regression


import torch 
import torch.nn as nn 
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# prepare data 

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target


n_samples, n_features = X.shape
print(n_samples, n_features)

X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#  scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_train = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32)) ## to tensor
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# reshape y tensor
y_train = y_train.view((y_train.shape[0], 1))
y_test = y_test.view((y_test.shape[0], 1))
# print(y_train.shape, y_test.shape)
## model

class model(nn.Module):
    def __init__(self, n_input_features):
        super(model,self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred= torch.sigmoid(self.linear(x))
        # print(y.shape)
        return y_pred


model =  model(n_features)


# loss function
lr = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

## traniing loop


n_epoch = 100

for epoch in range(n_epoch):

    # fwd pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # backward pass
    loss.backward()

    # update 
    optimizer.step()

    # zero gradients 
    optimizer.zero_grad()


    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss: {loss.item():.4f}')


with torch.no_grad():
    y_pred = model(X_test)
    y_pred_cls = y_pred.round()
    acc = y_pred_cls.eq(y_test.sum() / float(y_test.shape[0]))
    print(f'accuracy : {acc:.4f}')
     