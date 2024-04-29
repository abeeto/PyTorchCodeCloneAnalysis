import torch.nn as nn
from abc import abstractmethod


class DeepNetwork(nn.Module):
    def __init__(self):
        super(DeepNetwork, self).__init__()

        self.layers = self.initModelLayers()
        # convert layers list to sequential model(s)
        self.toSequential()
        # count model parameters
        self.nParams = sum([l.nelement() for l in self.parameters()])
        # layers list is no longer necessary as it has been converted to sequential model(s)
        del self.layers

    def className(self):
        return self.__class__.__name__

    @abstractmethod
    # init model layers as list of layers
    # returned list might be list of lists, if we init more than a single model
    def initModelLayers(self):
        raise NotImplementedError('subclasses must override initNNlayers()!')

    @abstractmethod
    # convert layers list to sequential model
    def toSequential(self):
        raise NotImplementedError('subclasses must override toSequential()!')

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('subclasses must override forward()!')

    # copy layers parameters values from given object
    def copyParams(self, srcObj):
        for v1, v2 in zip(srcObj.parameters(), self.parameters()):
            v1.data.copy_(v2.data)
