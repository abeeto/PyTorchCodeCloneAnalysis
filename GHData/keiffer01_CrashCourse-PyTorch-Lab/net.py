import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        '''
        You will build your own neural network here!
        Look back at the slides for some inspiration!

        Recap of Layers:
        - nn.Linear(in, out):
            a fully-connected layer that something of size "in" and returns
            something of size "out"
        - nn.Conv2d(in_channels, out_channels, kernel_size):
            convolutional layer that works on 2d images with number of channels
            given by "in_channels". Applies a number of filters given by
            "out_channels", with a kernel_size that can be an integer if square,
            or a tuple.
        - nn.MaxPool2d(kernel_size):
            pooling layer applied after the Conv2d layer. A larger kernel size
            results in a smaller remapped image.

        Some activation functions:
        - nn.Sigmoid(): applies sigmoid activation function (maps to range (0, 1))
        - nn.ReLU(): applies Rectified Linear Unit function (maps to (0, max(input)))
        - nn.Softmax(dim): applies softmax across the given dimension. Often
                           dim=1 if applied after a linear layer!

        Things to keep in mind:
        - The MNIST dataset was in black and white, so only had 1 channel. The
          CIFAR dataset is in RGB, so the input will have 3 channels.
        - Since we're doing a classification problem with 10 classes, make sure
          the final layer consists of a Linear layer with output 10, along with
          a Softmax activation function.
        '''
        super(MyNet, self).__init__()
        ### YOUR CODE HERE ###


    def forward(self, x):
        '''
        You will define the feedforward function here! Make use of the layers
        and activation functions you definied in __init__, and feed x through
        all of them. Return the transformed x.

        Things to keep in mind:
        - If your first layer is a Linear layer, make sure to flatten the input
          first! Since each image is a 32x32 image with 3 channels, you can
          do this with the function x.view(-1, 3 * 32 * 32).
        - Anytime you got from a Convolutional layer to a Linear layer, make
          sure to flatten the input to the Linear Layer first.
        '''
        ### YOUR CODE HERE ###
