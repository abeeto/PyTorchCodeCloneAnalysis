import torch
from torch.nn import ModuleList
import pickle
from utils import selectiveSearch, intersectionOverUnion, parseConfig

class EmptyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()



def makeModules(blocks):

    net_info =  blocks[0]
    blocks = blocks[1:] # remove net info
    modules = ModuleList()

    input_filters = 3 # 3 channel rgb
    output_filters = []

    for index, block in enumerate(blocks):
        module = torch.nn.Sequential()

        if block["name"] == "convolutional":
            batch_normalize = int(block["batch_normalize"])
            filters = int(block["filters"])
            size = int(block["size"])
            stride = int(block["stride"])
            pad = int(block["pad"])
            activation = block["activation"]
            pad = (size - 1) // 2 

            conv = torch.nn.Conv2d(input_filters, filters, size, stride, pad)
            module.add_module(f"conv_{index}", conv)

            if batch_normalize:
                batch_norm = torch.nn.BatchNorm2d(filters)
                module.add_module(f"batch_norm_{index}", batch_norm)

            if activation == "leaky":
                activation = torch.nn.LeakyReLU(inplace=True)
                module.add_module(f"leaky_{index}", activation)

        elif block["name"] == "upsample":
            stride = int(block["stride"])
            upsample = torch.nn.Upsample(scale_factor=stride, mode="bilinear")
            module.add_module(f"upsample_{index}", upsample)
        
        elif block["name"] == "route":
            layers = block["layers"].split(',').strip()

            if len(layers) > 1:
                start, end = layers[0], layers[1]
            else:
                start, end = layers[0], 0

            if start > 0:
                start -= index

            if end > 0:
                end -= index

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]


            route = EmptyLayer()
            module.add_module(f"route_{index}", route)

        elif block["name"] == "shortcut":
            from_ = int(block["from"])
            shortcut = EmptyLayer()
            module.add_module(f"shortcut_{index}", shortcut)

        elif block["name"] == "maxpool":
            stride = int(x["stride"])
            size = int(x["size"])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)

            module.add_module("maxpool_{}".format(index), maxpool)






class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        

