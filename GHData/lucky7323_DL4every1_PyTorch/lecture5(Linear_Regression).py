import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1) # one in, one out: 1st dim

    def forward(self, x):
        '''
        :param x: Variable
        :return: Variable
        '''
        y_pred = self.linear(x)
        return y_pred

model = Model()

# loss function and optimizer 
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training loop
for epoch in range(100):
    # forward pass
    y_pred = model(x_data)

    # compute loss
    loss = criterion(y_pred, y_data)
    print("epoch: %d, loss: %.3f" %(epoch, loss))

    # Zero gradients, Perform backward, Update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Because our model accepts only Variable,
# we must make our prediction input as a Variable
prediction = Variable(torch.Tensor([4.0]))
print("prediction of 4 hours: %.2f" % model(prediction))