# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset

###########################################################################################################
# A simple tutorial to demonstrate the basics for creating/train an MLP
# with single batch, mini-batch without DataLoader, and mini-batch with DataLoader.
# Part of the code is borrowed from
# https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
###########################################################################################################



# Two-layer MLP
class TwoLayerNet(torch.nn.Module):
    # Note that the model definition is no longer determined by the number of input samples per batch
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# single batch training over entire training set
def single_batch_demo_without_dataloader(N, D_in, H, D_out, train_x, train_y, test_x):
    # Construct our model by instantiating the class defined above
    model = TwoLayerNet(D_in, H, D_out)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    for epoch in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(train_x)

        # Compute and print loss
        loss = criterion(y_pred, train_y)
        if epoch % 50 == 49:
            print("epoch {}, loss {}".format(epoch + 1, loss.item()))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_out = model(test_x)
    print('test output:')
    print(y_out)


# mini-batch training, single batch pred
def mini_batch_demo_without_dataloader(N, D_in, H, D_out, batch_size, train_x, train_y, test_x):
    # Construct our model by instantiating the class defined above
    model = TwoLayerNet(D_in, H, D_out)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    for epoch in range(500):
        i = 0
        avg_loss = 0
        while i < N:
            batch_x = train_x[i: min(i + batch_size, N)]
            batch_y = train_y[i: min(i + batch_size, N)]

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(batch_x)

            # Compute and print loss
            loss = criterion(y_pred, batch_y)
            #print('loss', loss.item())
            avg_loss += loss

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += batch_size

        if epoch % 50 == 49:
            print("epoch {}, loss {}".format(epoch + 1, avg_loss.item() / np.ceil(N / batch_size)))

    y_out = model(test_x)
    print('test output:')
    print(y_out)


# mini-batch training, single batch pred with DataSet and DataLoader
def mini_batch_demo_with_dataloader(N, D_in, H, D_out, batch_size, train_x, train_y, test_x):

    # class MyDataset(Dataset):
    #     def __init__(self, x_tensor, y_tensor):
    #         self.x = x_tensor
    #         self.y = y_tensor
    #
    #     def __getitem__(self, index):
    #         return (self.x[index], self.y[index])
    #
    #     def __len__(self):
    #         return len(self.x)
    #
    #
    # train_data = MyDataset(train_x, train_y)
    # print(train_data[0])

    # with simple datasets this is effectively equivalent to using the custom dataset above
    train_data = TensorDataset(train_x, train_y)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    # Construct our model by instantiating the class defined above
    model = TwoLayerNet(D_in, H, D_out)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    for epoch in range(500):
        avg_loss = 0
        for batch_x, batch_y in train_loader:

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(batch_x)

            # Compute and print loss
            loss = criterion(y_pred, batch_y)
            #print('loss', loss.item())
            avg_loss += loss

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 50 == 49:
            print("epoch {}, loss {}".format(epoch + 1, avg_loss.item() / np.ceil(N / batch_size)))

    y_out = model(test_x)
    print('test output:')
    print(y_out)



if __name__ == '__main__':
    N, D_in, H, D_out, batch_size = 2000, 256, 128, 32, 100

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.

    # Create random Tensors to hold training inputs and outputs and test inputs
    train_x = torch.randn(N, D_in)
    train_y = torch.randn(N, D_out)

    test_x = torch.randn(N, D_in)

    single_batch_demo_without_dataloader(N, D_in, H, D_out, train_x, train_y, test_x)
    mini_batch_demo_without_dataloader(N, D_in, H, D_out, batch_size, train_x, train_y, test_x)
    mini_batch_demo_with_dataloader(N, D_in, H, D_out, batch_size, train_x, train_y, test_x)
