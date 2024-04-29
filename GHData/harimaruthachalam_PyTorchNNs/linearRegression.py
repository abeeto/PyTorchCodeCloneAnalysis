import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plot

len = 18
input_dim = 1
output_dim = 1
epochs = 10000

x_values = np.arange(len)
x_train = np.array(x_values, dtype=np.float32)

y_values = np.random.rand(len) * 5 + x_train * 2
y_train = np.array(y_values, dtype=np.float32)

x_train = x_train.reshape(-1,input_dim)
y_train = y_train.reshape(-1,output_dim)

'''
CREATE MODEL CLASS
'''
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

'''
INSTANTIATE MODEL CLASS
'''

model = LinearRegressionModel(input_dim, output_dim)

#######################
#  USE GPU FOR MODEL  #
#######################
if torch.cuda.is_available():
    model.cuda()

'''
INSTANTIATE LOSS CLASS
'''

criterion = nn.MSELoss()

'''
INSTANTIATE OPTIMIZER CLASS
'''

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


'''
TRAIN THE MODEL
'''
lossList = np.zeros(epochs)
for epoch in range(epochs):
    # Convert numpy array to torch Variable

    #######################
    #  USE GPU FOR MODEL  #
    #######################
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x_train).cuda())
        labels = Variable(torch.from_numpy(y_train).cuda())
    else:
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))

    # Clear gradients w.r.t. parameters
    optimizer.zero_grad()

    # Forward to get output
    outputs = model(inputs)

    # Calculate Loss
    loss = criterion(outputs, labels)

    # Getting gradients w.r.t. parameters
    loss.backward()

    # Updating parameters
    optimizer.step()

    # Logging
    print('epoch {}, loss {}'.format(epoch, loss.data[0]))
    lossList[epoch] = loss.data[0]
    epoch += 1

if torch.cuda.is_available():
    inputs = Variable(torch.from_numpy(x_train).cuda(), requires_grad=False)
    predicted = model(inputs).cpu().data.numpy()
else:
    inputs = Variable(torch.from_numpy(x_train), requires_grad=False)
    predicted = model(inputs).data.numpy()

# plot.plot(x_train, y_train)
# plot.plot(x_train, predicted)
# plot.legend(['Original Data','Predicted Data'])
# plot.xlabel('X')
# plot.ylabel('Y')
# plot.show()
# plot.close()

plot.plot(lossList)
plot.legend(['MSELoss'])
plot.xlabel('Epochs')
plot.ylabel('MSELoss')
plot.show()
plot.close()
