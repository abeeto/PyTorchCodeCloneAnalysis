import torch
from torch.autograd import Variable
import numpy as np

# generate the data
A = np.array([[1.0, 2.0],[3.0, 4.0]])
B = np.array([[4.0, 3.0],[2.0, 1.0]])

def to_torch(x):
  x = Variable(torch.from_numpy(x)).type(torch.cuda.FloatTensor)
  return x

def gen_xy():
  x = np.random.rand(2)
  y = np.matmul(A,x) if np.sum(x) > 1.0 else np.matmul(B,x)
  return x, y

def gen_xy_batch():
  xs, ys = [], []
  for i in range(30):
    x,y = gen_xy()
    xs.append(x)
    ys.append(y)
  return np.array(xs), np.array(ys)

print (gen_xy())

n_hidden = 200

model = torch.nn.Sequential(
          torch.nn.Linear(2, n_hidden),
          torch.nn.ReLU(),
          torch.nn.Linear(n_hidden, n_hidden),
          torch.nn.ReLU(),
          torch.nn.Linear(n_hidden, 2),
        ).cuda()

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-3

for t in range(5000):
  x, y = gen_xy_batch()
  x = to_torch(x)
  y = to_torch(y)

  y_pred = model(x)

  # Compute and print loss. We pass Variables containing the predicted and true
  # values of y, and the loss function returns a Variable containing the loss.
  loss = loss_fn(y_pred, y)
  print(t, loss.data[0])
  
  # Zero the gradients before running the backward pass.
  model.zero_grad()

  # Backward pass: compute gradient of the loss with respect to all the learnable
  # parameters of the model. Internally, the parameters of each Module are stored
  # in Variables with requires_grad=True, so this call will compute gradients for
  # all learnable parameters in the model.
  loss.backward()

  # Update the weights using gradient descent. Each parameter is a Variable, so
  # we can access its data and gradients like we did before.
  for param in model.parameters():
    param.data -= learning_rate * param.grad.data

for i in range(100):
  print ("========================")
  x, y = gen_xy()
  print (x)

  print ("prediction ")
  print (model(to_torch(x)))
  print ("truth")
  print (y)
