import torch as th
import torch.nn.functional as F
# -----------------------------------------------------------
# The class definition of Logistic Regression
class LogisticRegression(th.nn.Module):
    # create an object for logistic regression
    def __init__(self, p):
        super(LogisticRegression, self).__init__()
        # create a linear layer from p dimensional input to 1 dimensional output
        self.layer = th.nn.Linear(p, 1) # the linear layer includes parameters of weights (w) and bias (b)
    # the forward function, which is used to compute linear logits on a mini-batch of samples
    def forward(self, x):
        z = self.layer(x) # use the linear layer to compute z
        return z

# -----------------------------------------------------------
# The class definition of Convolutional Neural Network (CNN)
class CNN(th.nn.Module):
    # create an object for CNN 
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = th.nn.Conv2d( 1, 10, 5) # convolutional layer 1 
        self.conv2 = th.nn.Conv2d(10, 20, 5) # convolutional layer 2
        self.conv3 = th.nn.Conv2d(20, 30, 5) # convolutional layer 2
        self.fc = th.nn.Linear(30 * 4 * 4, 1) # linear layer
        self.pool = th.nn.MaxPool2d(2, 2) # max pooling layer
    # the forward function, which is used to compute linear logits in the last layer on a mini-batch of samples
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # convolutional layer 1
        x = self.pool(F.relu(self.conv2(x))) # convolutional layer 2
        x = self.pool(F.relu(self.conv3(x))) # convolutional layer 2
        x = x.view(-1, 30 * 4 * 4) # flattening
        z = self.fc(x) # linear layer
        return z

