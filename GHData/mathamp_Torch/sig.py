import torch
import torch.nn.functional as F


class MyModel(torch.nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0], [1.0]])

model = MyModel()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def learn(x, y):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.data.item()

"""
for epoch in range(1000):
    learn_loss = learn(x_data, y_data)
    print(f"{epoch} : {learn_loss}", end="\r")

"""