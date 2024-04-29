#Imports necessary to run program:
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader

#This Convolutional Neural Network Class that inherits from the pl.LightningModule.
#Uses PyTorch's Conv2d() functions followed by PyTorch's ReLU() functions followed by MaxPool2d() functions.
#Selected hyperparameters of in_channels, out_channels, kernel_size, and padding were chosen based on the equation for
#Convolutional Neural Networks: O = ((W-K+2P)/S)+1, where O is the output height/length, W is the input height/length,
#K is the filter size, P is the padding, and S is the stride.


class ConvolutionalNeuralNetwork(pl.LightningModule):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding = 1)
        self.r1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(kernel_size=3)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding = 1)
        self.r2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(kernel_size=2)
        self.conv1_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.r3 = nn.ReLU()
        self.mp3 = nn.MaxPool2d(kernel_size=3)
        self.linear_trans = nn.Linear(64*1*1, 10)
        self.accuracy = pl.metrics.Accuracy()

    # Forward function that grabs the transformed results from the constructor and then passes them to the next function.
    def forward(self, x):
        result = self.conv1_1(x)
        result = self.r1(result)
        result = self.mp1(result)
        result = self.conv1_2(result)
        result = self.r2(result)
        result = self.mp2(result)
        result = self.conv1_3(result)
        result = self.r3(result)
        result = self.mp3(result)
        result = result.view(result.size(0), -1)
        result = self.linear_trans(result)
        return result

    #This is the training_step necessary for PyTorch Lightning. Contains the loss function in loss, the activation
    #functions are called in output and makes a log of the information under the Lightning_logs folder of each run.
    #Returns the loss.

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = F.multi_margin_loss(output, y)
        self.log('Loss', loss)
        self.log('Accuracy Step', self.accuracy(output, y))
        return loss

    #Function that computes the accuracy and displays the progress bar of each epoch.
    def training_epoch_end(self, outputs):
        self.log('Accuracy Epoch', self.accuracy.compute(), prog_bar = True)
    #Function that selects the optimizer.
    #Returns the optimizer.
    def configure_optimizers(self, ):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer

if __name__ == '__main__':

    #You have to replace the directory in image_data with the directory where the downloaded image dataset is on
    #your computer.

    image_data = "/home/thomas/notMNIST_small"

    #Load and transform
    trsfrm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,), (1.0,))])

    training_set = datasets.MNIST(root = image_data, train = True, transform = trsfrm, download = True)
    testing_set = datasets.MNIST(root = image_data, train = False, transform = trsfrm, download = True)

    batch_size = 64

    training_loader = torch.utils.data.DataLoader(dataset = training_set, batch_size = batch_size, shuffle = True)
    testing_loader = torch.utils.data.DataLoader(dataset = testing_set, batch_size = batch_size, shuffle = True)

    #Instantiate model, specify epochs, and fit.
    cnn = ConvolutionalNeuralNetwork()
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(cnn, training_loader, testing_loader)