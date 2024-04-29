
import torch.nn as nn
from collections import OrderedDict
class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        self.body = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)),#params 5*5*1*6 = 150
            ('relu1', nn.ReLU()),
            ('max_pool_1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)),#params 5*5*6*16 = 2400
            ('relu2', nn.ReLU()),
            ('max_pool_2', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

        self.head = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(16 * 5 * 5, 120)), #params =48000
            # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
            ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(120, 84)), #params = 10080
            # convert matrix with 120 features to a matrix of 84 features (columns)
            ('relu4', nn.ReLU()),
            ('fc3', nn.Linear(84, 10)), #params = 840
            ('softmax',nn.Softmax())
        ]))

        ##params total = 150+2400+48000+10080+840=61470

    def forward(self, img):
        img = self.body(img)
        img = img.view(-1, 16 * 5 * 5)
        output = self.head(img)
        return output


class LeNet5_regulized(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super(LeNet5_regulized, self).__init__()

        self.body = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)),#params 5*5*1*6 = 150
            ('relu1', nn.ReLU()),
            ('max_pool_1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)),#params 5*5*6*16 = 2400
            ('relu2', nn.ReLU()),
            ('max_pool_2', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

        self.head = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(16 * 5 * 5, 120)), #params =48000
            ('dropout', nn.Dropout(0.15)),
            # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
            ('relu3', nn.ReLU()),
            ('batnorm1', nn.BatchNorm1d(120)),
            ('fc2', nn.Linear(120, 84)), #params = 10080
            # convert matrix with 120 features to a matrix of 84 features (columns)
            ('relu4', nn.ReLU()),
            # ('batnorm2', nn.BatchNorm1d(84)),
            ('fc3', nn.Linear(84, 10)), #params = 840
            ('softmax',nn.Softmax())
        ]))

        ##params total = 150+2400+48000+10080+840=61470

    def forward(self, img):
        img = self.body(img)
        img = img.view(-1, 16 * 5 * 5)
        output = self.head(img)
        return output


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5

        --  custom mlp model should be just nn.linear module? or can use conv and maxpooling?
        -- then mlp model will be have  poor performance
    """

        # write your codes here
    def __init__(self):
        super(CustomMLP, self).__init__()


        self.body = nn.Sequential(OrderedDict([
            ('FC1', nn.Linear(28*28*1,60)), #params 23520, 61470-47040 = 14430
            ('relu1', nn.ReLU()),
            ('FC2', nn.Linear(60,60)),      #params 3600  , 14430-3600 = 10830
            ('relu2', nn.ReLU()),
            ('FC3', nn.Linear(60,60)),      #params 3600  , 10830-3600 = 7,230
            ('relu2', nn.ReLU()),
            ('FC4', nn.Linear(60,60)),      #params 3600  , 7,230-3600 = 3630
            ('relu2', nn.ReLU()),
            ('FC5', nn.Linear(60, 52)),     # params 3120  , 3630-3120 = 510
            ('relu2', nn.ReLU()),
            ('FC6', nn.Linear(52, 10)),     # params 520  , 510-520 = -10
            ('softmax', nn.Softmax())
        ]))



    def forward(self, img):

        img = img.view(-1, 28*28*1)
        output = self.body(img)

        return output

