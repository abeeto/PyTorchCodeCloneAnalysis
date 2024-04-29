import torch as torch
import torch.nn.functional as functional
from torch.autograd import Variable
import numpy as np

x_data = np.array(
    [[73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58],
     [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70]],
    dtype='float32')
y_data = np.array([[56, 70], [81, 101], [119, 133], [22, 37], [103, 119],
                   [56, 70], [81, 101], [119, 133], [22, 37], [103, 119],
                   [56, 70], [81, 101], [119, 133], [22, 37], [103, 119]], dtype='float32')

x_data = Variable(torch.from_numpy(x_data))
y_data = Variable(torch.from_numpy(y_data))


class multivarModel(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(multivarModel, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = functional.relu(self.hidden(x))
        x = self.predict(x)
        return x

    # our model


our_model = multivarModel(n_feature=3, n_hidden=200, n_output=2)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(our_model.parameters(), lr=0.05)

for epoch in range(500):
    # Forward pass: Compute predicted y by passing
    # x to the model
    pred_y = our_model(x_data)

    # Compute and print loss
    loss = criterion(pred_y, y_data)

    # Zero gradients, perform a backward pass,
    # and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))

new_var = Variable(torch.Tensor([[73, 67, 43]]))
pred_y = our_model(new_var)
print("predict (after training)", [73, 67, 43], our_model(new_var))
