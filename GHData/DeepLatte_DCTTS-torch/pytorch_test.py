import torch
from torch.autograd import Variable
import torch.nn.functional as F

x_data = Variable(torch.Tensor(([1.0], [2.0], [3.0], [4.0])))
y_data = Variable(torch.Tensor([[0.],[0.],[1.],[1.]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = Model()

criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(torch.cuda.is_available())
print('done')