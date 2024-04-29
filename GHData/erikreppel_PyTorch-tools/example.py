from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiment import Experiment


# Use FashionMNIST as a dataset
class FashionMNIST(MNIST):
    '''Implement Dataset with FashionMnist'''
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]

# define image transforms
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

trainset = FashionMNIST('./data', download=True, transform=transform)
testset = FashionMNIST('./data', train=False, download=True, transform=transform)

batch_size = 64

train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)


# Basic convnet
class FashionModel(nn.Module):
    def __init__(self):
        super(FashionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

from experiment import Experiment

# Create the model and experiment
model = FashionModel()
exp = Experiment(model)

# define the loss and optimization functions
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# compile the experiment
exp.compile(optimizer, criterion)

# train and test
exp.fit(train_loader, n_epoch=10)

exp.evaluate(test_loader)
