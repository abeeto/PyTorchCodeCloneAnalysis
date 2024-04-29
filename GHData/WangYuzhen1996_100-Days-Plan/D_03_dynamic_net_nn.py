#--------------------------------------------------#
# The code referenced from below                   #
# https://github.com/jcjohnson/pytorch-examples    #
#--------------------------------------------------#

#===========================================#
#                 Sample 1                  #
#              dynamic_net_nn               #
#                  python3                  #
#===========================================#
# Code in file dynamic_net_nn.py
import torch
import random

class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        :param D_in: input dimension
        :param H: hidden dimension
        :param D_out: output dimension
        In the constructor we construct three nn.Linear instances that we will use
        in the forward passs.
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear =  torch.nn.Linear(H, D_out)

    def forward(self,x):
        """
        :param x: forward pass
        :return:  y_pred
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):           # Dynamic graph
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred
# N is batch size; D_in is input dimension
# H is hidden dimension; D_out is output dimension
N,D_in,H,D_out = 64, 1000,100,10
# Create random Tensors
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
# The model
model = DynamicNet(D_in, H, D_out)

# Loss and optimizer
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(),lr=1e-4,momentum=0.9)

# Training
for t in range(500):
    # Forward pass
    y_pred = model(x)

    # Compute and print loss
    loss=criterion(y_pred, y)
    print(t,loss.item())

    # Zero gradients, perform a backward pass, and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
