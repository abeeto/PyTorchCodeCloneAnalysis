"""
PyTorch-MNIST.py
    MNIST model using PyTorch
Bryce Harrington
09/16/22
"""
# base torch dependencies
import torch
import torchvision.transforms
from torch import nn, flatten
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader

# torchvision data / functions
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST

# our config file
from Config import LEARNING_RATE, BATCH_SIZE, EPOCHS, TRAIN_SPLIT

# other necessary packages
import numpy as np
from sklearn.metrics import classification_report


# define our model as a class object
class Network(nn.Module):
    def __init__(self, channels: int, classes: int):
        """
        Define network layers for a LeNet implementation
        """
        # call module constructor
        super(Network, self).__init__()

        # first convolutional layers
        self.conv_1 = nn.Conv2d(in_channels=channels, out_channels=20, kernel_size=(5, 5))
        self.relu_1 = nn.ReLU()
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # second convolutional layers
        self.conv_2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu_2 = nn.ReLU()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # fully connected layers
        self.fc_1 = nn.Linear(in_features=800, out_features=500)
        self.relu_3 = nn.ReLU()

        # classification layer
        self.fc_2 = nn.Linear(in_features=500, out_features=classes)
        self.softmax = nn.LogSoftmax(dim=channels)

    def forward(self, input_t):
        """
        The structure and order of our network layers
        :param input_t: the input tensor
        :return: output: the output prediction from softmax
        """
        # pass input tensor to our input conv layer
        input_t = self.conv_1(input_t)
        input_t = self.relu_1(input_t)
        input_t = self.max_pool_1(input_t)

        # pass output of first layer through second conv layer
        input_t = self.conv_2(input_t)
        input_t = self.relu_2(input_t)
        input_t = self.max_pool_2(input_t)

        # pass output of second layer through our fully connected layers ( so we can converge on an output )
        input_t = flatten(input_t, 1)
        input_t = self.fc_1(input_t)
        input_t = self.relu_3(input_t)

        # pass our fully connected output to a final fc layer and predict with softmax
        input_t = self.fc_2(input_t)
        output = self.softmax(input_t)

        return output


def train():
    # set our device for PyTorch to run on
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load our dataset ( KMNIST )
    train_data = KMNIST(root="data", train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_data = KMNIST(root="data", train=False, download=True, transform=torchvision.transforms.ToTensor())

    # split our data into our test, train and validation sets
    train_split = int(len(train_data)*TRAIN_SPLIT)
    (train_data, val_data) = random_split(train_data,
                                          [train_split, int(len(train_data) - train_split)],
                                          generator=torch.Generator())

    # place our data into our torch data loaders
    o_test_data = test_data
    test_data = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)
    train_data = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    val_data = DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE)

    # init our previously defined model
    model = Network(channels=1, classes=len(train_data.dataset.dataset.classes)).to(compute_device)

    # set out optimizer and loss
    model_opt = Adam(model.parameters(), lr=LEARNING_RATE)
    model_loss = nn.NLLLoss()

    # begin training
    for epoch in range(EPOCHS):
        # put model into a training state
        model.train()

        # init our loss values
        train_loss, val_loss = 0, 0
        # init our accuracy tracking values
        train_correct, val_correct = 0, 0

        # go through our training data
        for (data, label) in train_data:
            # prefetch batched data to compute device
            (data, label) = (data.to(compute_device), label.to(compute_device))

            # pass data through the network
            output = model(data)
            loss = model_loss(output, label)

            # ZERO THE GRADIENTS ( it's always a blast when we forget too )
            model_opt.zero_grad()
            loss.backward()
            model_opt.step()

            # update our train loss values
            train_loss += loss
            train_correct += (output.argmax(1) == label).type(torch.float).sum().item()

        # assess the performance of the model
        with torch.no_grad():
            # eval mode to be sure it doesn't learn our validation set ( since it would kind of defeat the point )
            model.eval()

            for (data, label) in val_data:
                # just like above, assign our data to the compute device
                (data, label) = (data.to(compute_device), label.to(compute_device))

                # again we predict, and grab loss
                output = model(data)
                loss = model_loss(output, label)

                # update our correct val predictions
                val_correct += (output.argmax(1) == label).type(torch.float).sum().item()
                val_loss += loss

        # compute validation / training loss
        train_loss /= (len(train_data.dataset) // BATCH_SIZE)
        val_loss /= (len(val_data.dataset) // BATCH_SIZE)

        # compute accuracy scores
        train_correct /= len(train_data.dataset)
        val_correct /= len(train_data.dataset)

        # display current model stats
        print(f"[INFO]: Epoch: {epoch}, Accuracy: {train_correct}, Loss: {train_loss}, Val_Accuracy: {val_correct}, Val_Loss: {val_loss}")

    # once that's done, lets verify our values with the test set
    with torch.no_grad():
        # eval mode
        model.eval()

        output = []
        for (data, label) in test_data:
            # once more, make sure it's sent to our compute device of choice
            data = data.to(compute_device)

            # save our output to, well 'output' above
            outputs = model(data)
            output.extend(outputs.argmax(axis=1).cpu().numpy())

    # generate and output classification report
    print(classification_report(o_test_data.targets.cpu().numpy(), np.array(output), target_names=o_test_data.classes))


if __name__ == "__main__":
    train()
