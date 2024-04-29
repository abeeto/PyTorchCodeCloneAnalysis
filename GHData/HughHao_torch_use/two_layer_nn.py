# -*- coding: utf-8 -*-
# @Time : 2022/4/10 16:40
# @Author : hhq
# @File : two_layer_nn.py
# todo 用NeuralNetwork库实现双层神经网络
import torch

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
# Use the nn package to define our model as a sequence of layers. nn.Sequential\n",
# is a Module which contains other Modules, and applies them in sequence to\n",
# produce its output. Each Linear Module computes output from input using a\n",
# linear function, and holds internal Tensors for its weight and bias.\n",
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)
##将model里面的第一层和第三层的权重初始化为正态分布
torch.nn.init.normal_(model[0].weight)
torch.nn.init.normal_(model[2].weight)
# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-6

for t in range(500):
    # Forward pass: compute predicted y by passing x to the model. Module objects\n",
    # override the __call__ operator so you can call them like functions. When\n",
    # doing so you pass a Tensor of input data to the Module and it produces\n",
    # a Tensor of output data.\n",
    y_pred = model(x)
    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the loss.,
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    # Zero the gradients before running the backward pass.\n",
    model.zero_grad()
    loss.backward()
    # Update the weights using gradient descent. Each parameter is a Tensor, so\n",
    # we can access its gradients like we did before.\n",
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
