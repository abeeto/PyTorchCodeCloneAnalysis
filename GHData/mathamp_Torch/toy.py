import torch
import torch.nn.functional as F


class MyModule(torch.nn.Module):

    def __init__(self):
        super(MyModule, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return F.ReLU(self.linear(x))


x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0], [1.0]])

model = MyModule()
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(5000):

    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(f"{epoch} epoch loss : {loss.data.item()}", end="\r")
    optimizer.zero_grad()
    loss.backward()     # 편미분 구하기. f가 정해져야 편미분 가능하지.    d(loss)/d(var)
    optimizer.step()    # 편미분으로 적용.

for i in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
    print(f"{i} hour : {model(torch.tensor([i])).data[0].item()}")

