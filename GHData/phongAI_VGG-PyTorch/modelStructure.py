import torch
import torch.nn as nn
from torch import optim
import time
from tqdm import tqdm

class VGG(nn.Module):
    architecture = [64,64,"MAX",128,128,"MAX",256,256,256,"MAX",512,512,512,"MAX",512,512,512,"MAX"]

    def __init__(self,in_chanel = 3,input_shape = 224,classes = 1000):
        super(VGG, self).__init__()
        self.in_chanel = in_chanel
        self.input_shape = input_shape
        self.classes = classes
        #Feature extractor
        self.feature_extractor_layers = self.create_feature_extractor()

        #Full Connected
        self.fcs = self.create_full_connected()

        #Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def create_feature_extractor(self):
        sequeltial = []
        self.downscale = 1
        for layer in self.architecture:
            if type(layer) == int:
                out_feature = layer
                sequeltial += [nn.Conv2d(in_channels=self.in_chanel,out_channels=out_feature
                                         ,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                               nn.BatchNorm2d(out_feature),
                               nn.ReLU()
                               ]
                self.in_chanel = out_feature
            elif layer == "MAX":
                kernel_size = 2
                sequeltial += [nn.MaxPool2d(kernel_size=kernel_size,stride=(2,2))]
                self.downscale *= kernel_size
        return nn.Sequential(*sequeltial)

    def create_full_connected(self):
        sequential = nn.Sequential(
            nn.Linear(in_features=self.architecture[-2] * int(self.input_shape / self.downscale) ** 2,
                      out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=self.classes),
        )
        return sequential

    def forward(self,x):
        #Forward data to extractor structure
        x = self.feature_extractor_layers(x)
        #Flatten
        x = torch.flatten(x,1)
        #Go through full-connected layers
        x = self.fcs(x)
        return x

    def _saveCheckpoint(self,checkpoint,fileName = "sample.pth.tar"):
        torch.save(checkpoint,fileName)

    def _loadCheckpoint(self,checkpoint):
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


    def batch_train(self,input_data = None,one_hot =False,lr=1e-3,epochs = 10):
        print("Start training")
        """Optimizer"""
        self.optimizer = optim.Adam(self.parameters(),lr = lr)
        lossFunction = nn.CrossEntropyLoss()

        checkpoint = {'state_dict':self.state_dict(),'optimizer:':self.optimizer.state_dict()}
        """Train"""
        if input_data != None:
            for epoch in range(epochs):
                if epoch % 1 == 0:
                    self._saveCheckpoint(checkpoint)
                loop = tqdm(enumerate(input_data),total=len(input_data),leave=False)
                for index,(imageData,label) in loop:
                    self.optimizer.zero_grad()
                    imageData = imageData.cuda()
                    label = label.cuda()
                    output = self.forward(imageData)
                    loss = lossFunction(output,label)
                    loss.backward()
                    self.optimizer.step()
                    loop.set_description(f"Epoch [{epoch}/{epochs}]")
                    loop.set_postfix(loss = loss.item(),acc = torch.rand(1).item())





