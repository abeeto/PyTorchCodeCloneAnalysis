import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F


# hyperparameters
batch_size = 5
input_size = 3
hidden_size = 4
num_classes = 5
learning_rate = 0.001
num_epoch = 1000

# manual_seed sets the random seed from pytorch random number generators
torch.manual_seed(123)

# input
input = Variable(torch.rand(batch_size, input_size))
# print('input = ', input)

# output for training
target = Variable((torch.rand(batch_size)*num_classes).long())
print('target = ', str(target.view(1, -1)).split('\n')[1])


# Model Definition
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.h1(x)
        x = F.tanh(x)
        x = self.h2(x)
        x = F.softmax(x, dim=0)
        return x


# calling the model
model = Net(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

# optimizer (in this case we use: ADAM)
opt = optim.Adam(params=model.parameters(), lr=learning_rate)


# Learning process
for epoch in range(num_epoch):

    # print('######## EPOCH:', epoch, '########')

    # forward path
    out = model(input)
    # print('out = ', out)

    # max over the axis 1, which is max over the average raw.
    # the output of the max is: maximum value, index of the maximum value
    _, pred = out.max(1)
    if epoch % 100 == 0:
        print('prediction = ', str(pred.view(1, -1)).split('\n')[1])

    # loss function
    loss = F.nll_loss(out, target)
    if epoch % 100 == 0:
        print('loss = ', loss.data[0])

    # zero the gradient inside the both of the linear layers
    model.zero_grad()

    # calculate the gradient
    loss.backward()

    # make a step towards good solution
    opt.step()
