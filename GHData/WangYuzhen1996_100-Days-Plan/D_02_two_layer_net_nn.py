#--------------------------------------------------#
# The code referenced from below                   #
# https://github.com/jcjohnson/pytorch-examples    #
#--------------------------------------------------#

#===========================================#
#                 Sample 1                  #
#              two_layer_net_nn             #
#                  python3                  #
#===========================================#
# Code in file two_layer_net_nn.py
import torch

device = torch.device('cpu')
# device = torch.device('cuda'0
# N is batch size; D_in is input dimension
# H is hidden dimension; D_out is output dimension
N,D_in,H,D_out = 64, 1000,100,10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module compute output from input using a
# Linear function,and holds interal Tensors for its weight and bias.
# After constructing the model we use thr .to() method to move it to the
# desired deviced
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)

).to(device)
# The nn package also contain definitions of popular loss function; in this
# Case we will use Mean Squared Error(MSE) as our function
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4

for t in range(500):
    # Forward pass : compute predicted y by passing x to the model. Module objects
    # override the __call__operator so you can call them like functions. When
    # doing so you pass a Tensors of input data to the Module and it produces
    # a tensor of output data
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function return a Tensor contain the loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Zero the gradients before running the backward pass
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Interally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using the gradients. Each parameter is a Tensor, so
    # we can access its data and gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param.data -= learning_rate * param.grad
