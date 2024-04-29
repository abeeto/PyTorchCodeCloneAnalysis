# Lab Ex12-1
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Training settings


######################################################################
# Network Outputs: 101333 or 202333				     #
#                  ihilll or ehelll				     #
######################################################################

######################################################################
# The input of "h" occurs for both the outputs of "i" and "e"	     #
#								     #
# Order is important - Linear classifier does not recognize this     #
#								     #
# Additionally occurs with input "l" in the last position of "o"     #
#								     #
# Both occur equally, however, "l" is a more stable guess due to its #
# more frequent output occurence				     #
######################################################################	


#Generate 1-hot encoding of input values	
x = [[0, 1, 0, 2, 3, 3]]
x_one_hot = [[1, 0, 0, 0, 0],
		[0, 1, 0, 0, 0],
		[1, 0, 0, 0, 0],
		[0, 0, 1, 0, 0],
		[0, 0, 0, 1, 0],
		[0, 0, 0, 1, 0]]

y = [1, 0, 2, 3, 3, 4]


#Attempt to create linear function to estimate the output
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x)


#Initialize
model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

#Train
def train(epoch):
    model.train()
    inputs = Variable(torch.Tensor(x_one_hot))
    labels = Variable(torch.LongTensor(y))
    for i in range(100):
        optimizer.zero_grad()
        output = model(inputs)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(output.data.max(1)[1])


for epoch in range(1, 10):
    train(epoch)

