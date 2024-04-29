import torch
import random

device = torch.device("cpu")
# device = torch.device("cuda:0") # uncomment this line to run on GPU

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor, we instantiate two nn.Linear modules
        and assign them as member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return a tensor
        of output data. We can use Modules defined in the constructor as well as arbitrary
        operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we construct 3 Linear Modules that we will use in the forward pass.
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose a number from 0 to 3 and reuse
        the middle module that many times to compute hidden layer representations.
        Since each forward pass build a dynamic computation graph, we can use normal Python
        control-flow operators like loops and conditional statements when defining the formal pass.
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


# N is batch size; D_in is input dimension; H is hidden dimension; D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# create random Tensors to hold input and output data
x = torch.randn(N, D_in, device=device, dtype=torch.float)
y = torch.randn(N, D_out, device=device, dtype=torch.float)

#model = TwoLayerNet(D_in, H, D_out)
model = DynamicNet(D_in, H, D_out)

# we use Mean Squared Error as our loss function (from the nn package)
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
# the SGD optimizer will update the weights of the model. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the 2 linear Modules of the model
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) # add momentum = 0.9 for DynamicNet

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

    # calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()