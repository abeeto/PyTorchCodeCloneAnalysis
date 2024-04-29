import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Class definition of the model we will be using in the training.
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

        self.checkpoint_dir = './chkpt/'

    def forward(self, tens):
        """
        input tensor forward propagation implementation.
        """
        # hidden conv1 layer
        tens = self.conv1(tens)
        tens = F.relu(tens)
        tens = F.max_pool2d(tens, kernel_size=2, stride=2)

        # hidden conv2 layer
        tens = self.conv2(tens)
        tens = F.relu(tens)
        tens = F.max_pool2d(tens, kernel_size=2, stride=2)

        # hidden fc1 layer
        # flatten the output from previous conv layers
        tens = tens.reshape(-1, 12 * 4 * 4)
        tens = self.fc1(tens)
        tens = F.relu(tens)

        # hidden fc2 layer
        tens = self.fc2(tens)
        tens = F.relu(tens)

        # output layer
        # NB: output layer does not get activated because it will be later in the loss computation
        tens = self.out(tens)

        return tens

    def save(self, run):
        torch.save(self.state_dict(), self.checkpoint_dir + str(run))

    def load(self, run):
        checkpoint = torch.load(self.checkpoint_dir + str(run))
        self.load_state_dict(checkpoint)
