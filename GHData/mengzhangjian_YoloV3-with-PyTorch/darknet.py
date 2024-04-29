from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F   
from torch.autograd import Variable
import numpy as np   

def parse_cfg(cfgfile):
    """
    Takes as configuration file

    Returns a list of blocks.Each blocks describes a block in the neural network to be built.import
    Blocks is represented as a dictionary in the list
    """

    file = open(cfgfile,'r')
    lines = file.read().split('\n')  #store the lines in a list
    lines = [x for x in lines if len(x) > 0 ]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) !=0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split('=')
            block[key.rstrip()]  = value.lstrip()
    blocks.append(block)

    return blocks
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer,self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self,anchors):
        super(DetectionLayer,self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    net_info = blocks[0] #Caputures the information about the input and pre-processing
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        #check the type of block,create a new module for the block
        #append to module_list

        if(x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = True
            except:
                batch_normalize = 0
                bias = False
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1 )// 2
            else:
                pad = 0
            #Add the convolution layer
            conv = nn.Conv2d(prev_filters,filters,kernel_size,stride,pad,bias=bias)
            module.add_module("conv_{0}".format(index),conv)
            
            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index),bn)
            #Check the activation
            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1,inplace=True)
                module.add_module("leaky_{0}".format(index),activn)
        #If it's an upsampling layer
        #We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor= stride,mode = "bilinear")
            module.add_module("Upsample_{}".format(index),upsample)
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            #Start of a route
            start  = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation

            if start > 0:
                start = start -index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index),route)

            if end < 0 :
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
        #shortcut corresponds to skip connection

        elif x["type"] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index),shortcut)
        #YOLO is the detection layer
        
        elif x["type"] =="yolo":
            mask = x["mask"].split(',')
            mask = [int(temp) for temp in mask]
            anchors =  x["anchors"].split(',')
            print(anchors)
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index),detection)
        
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return (net_info,module_list)


        
                            
