import torch

device = torch.device("cpu")
# device = torch.device("cuda:0") # uncomment this line to run on GPU

# N is batch size; D_in is input dimension; H is hidden dimension; D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# create random Tensors to hold input and output data
x = torch.randn(N, D_in, device=device, dtype=torch.float)
y = torch.randn(N, D_out, device=device, dtype=torch.float)

# use the nn package to define our model as a sequence of layers. nn.Sequential is a Module
# which contains other Modules, and applies them in sequence to produce its output. Each
# Linear Module computes output from input using a linear function, and holds internal Tensors
# for weights and biases
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# we use Mean Squared Error as our loss function (from the nn package)
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
# the Adam optimizer will update the weights of the model.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    # forward pass: compute predicted y by passing x to the model. Module objects override
    # the __cal__ operator so you can call them like functions. When doing so you pass a
    # Tensor of input data to the Module and it produces a tensor of output data.
    y_pred = model(x)

    # compute and print loss. We pass Tensors containing the predicted and true values of y,
    # and the loss functions returns a Tensor containing the loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # zero the gradients before the backward pass
    optimizer.zero_grad()

    # backward pass: compute gradient of the loss with respect to all the learnable parameters
    # of the model. Internally, the parameters of each Module are stored in Tensors with
    # requires_grad=True, so this call will compute gradients for all learnable parameters in the model
    loss.backward()

    # update the weights using gradient descent. Each parameters is a Tensor, so we can access its
    # gradients like we did before.
    # with torch.no_grad():
    #     for param in model.parameters():
    #         param -= learning_rate * param.grad

    # calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()