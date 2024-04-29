
from torch import nn
from torchsummary import summary

class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        #4 conv blocks / flatten layer  / linear layer / sfotmax

        """
        Convolutional blocks to extract features from input mel-spectrogram
        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        """
        Flatten layer
        """
        self.flatten = nn.Flatten()
        """
        Fully connected blocks to do classification
        """
        self.linear = nn.Linear(128 * 5 * 4, 10)
        """
        Softmax used to normalize results and obtain probabilities
        """
        self.softmax = nn.Softmax()


    def forward(self, input_data):
        x = self. conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions

if __name__ == "__main__":

    cnn = CNNNetwork()
    summary(cnn.cuda(), (1,64,44))