import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import *


def parse_cfg(file):
    lines = []
    with open(file, "r") as fin:
        lines = fin.readlines()
    layers = []
    layer = {}
    for l in lines:
        if len(l) <= 2:
            continue
        else:
            if l[0] == '[':
                if len(layer) > 0:
                    layers.append(layer)
                layer = {}
                layer["type"] = l[1:-2]
            elif l[0] == '#':
                continue
            else:
                key, val = l.split("=")
                layer[key.strip()] = val.strip()
    if len(layer) > 0:
        layers.append(layer)
    return layers



def create_modules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing    
    
    module_list = nn.ModuleList()
    
    index = 0    #indexing blocks helps with implementing route  layers (skip connections)

    
    prev_filters = 3
    
    output_filters = []
    
    for x in blocks:
        module = nn.Sequential()
        
        if (x["type"] == "net"):
            continue
        
        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
                
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
                
            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
            
            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
            
            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)
            
            
            
        #If it's an upsampling layer
        #We use Bilinear2dUpsampling
        
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
#            upsample = Upsample(stride)
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)
        
        #If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            
            #Start  of a route
            start = int(x["layers"][0])
            
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
                
            
            
            #Positive anotation
            if start > 0: 
                start = start - index
            
            if end > 0:
                end = end - index

            
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            
            
            
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]
                        
            
        
        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            from_ = int(x["from"])
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
            
            
        elif x["type"] == "maxpool":
            stride = int(x["stride"])
            size = int(x["size"])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)
            
            module.add_module("maxpool_{}".format(index), maxpool)
        
        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
            
            
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
            
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
        
            
            
        else:
            print("Unkown layer", x["type"])
            assert False


        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index += 1
        
    
    return net_info, module_list




class EmptyLayer(nn.Module):
    
    def __init__(self):
        super(EmptyLayer, self).__init__()
        

class DetectionLayer(nn.Module):
    
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        
    def forward(self, x, inp_dim, num_classes, confidence):
        x = x.data
        global CUDA
        prediction = x
        prediction = predict_transform(prediction, inp_dim, self.anchors, num_classes, confidence, CUDA)
        return prediction



class Yolo(nn.Module):
    
    def __init__(self, cfg_file):
        super(Yolo, self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        detections = []
        modules = self.blocks[1:]
        outputs = []   #We cache the outputs for the route layer
        
        for i, module in enumerate(modules):
            module_type = module["type"]
            
            if module_type == "convolutional" or \
               module_type == "upsample" or module_type == "maxpool":
                x = self.module_list[i](x)
                outputs.append(x)
                
            elif module_type == "route":
                layers = list(map(int, module["layers"]))
                
                # express layer[0] relative to the current layer                
                if layers[0] > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    # express layer[0] relative to the current layer                
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    # get the two layers results
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    # and concatenat them along depth
                    x = torch.cat((map1, map2), 1)
                outputs.append(x)
            
            elif  module_type == "shortcut":
                from_ = int(module["from"])
                # sum theprevious layer with another one specified by from_
                x = outputs[i - 1] + outputs[i + from_]
                outputs.append(x)
            
            elif module_type == 'yolo':       
                # final layer which transforms the results map into an array with one detection per line
                # WARNING: this layer can happen different time in the full network
                #          because different resolutions are processed
                anchors = self.module_list[i][0].anchors   # available anchors
                inp_dim = int(self.net_info["height"])     # input dimension
                num_classes = int (modules[i]["classes"])  # number of classes
                
                x = x.data   # get the Tensor under the Variable
                # organize the detection with one line per BBox
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                
                if x.numel() == 0:
                    continue

                detections.append(x)
                outputs.append(outputs[-1])
             
        
        if len(detections) > 0:
            return torch.cat(detections, 1)
        else:
            return 0
    
    
    def load_weights(self, file):
        with open(file, "rb") as fin:
            header = np.fromfile(fin, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            weights = np.fromfile(fin, dtype=np.float32)
            ptr = 0
            for i in range(len(self.module_list)):
                mod_type = self.blocks[i+1]["type"]
                
                if mod_type == "convolutional":
                    layer = self.module_list[i]
                    batch_normalize = False
                    if "batch_normalize" in self.blocks[i+1].keys():
                        batch_normalize = self.blocks[i+1]["batch_normalize"]

                    conv = layer[0]
                    
                    if (batch_normalize):
                        bn = layer[1]
            
                        #Get the number of weights of Batch Norm Layer
                        num_bn_biases = bn.bias.numel()
            
                        #Load the weights
                        bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                        ptr += num_bn_biases
            
                        bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr  += num_bn_biases
            
                        bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr  += num_bn_biases
            
                        bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr  += num_bn_biases
            
                        #Cast the loaded weights into dims of model weights. 
                        bn_biases = bn_biases.view_as(bn.bias.data)
                        bn_weights = bn_weights.view_as(bn.weight.data)
                        bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                        bn_running_var = bn_running_var.view_as(bn.running_var)
            
                        #Copy the data to model
                        bn.bias.data.copy_(bn_biases)
                        bn.weight.data.copy_(bn_weights)
                        bn.running_mean.copy_(bn_running_mean)
                        bn.running_var.copy_(bn_running_var)
                    else:
                        #Number of biases
                        num_biases = conv.bias.numel()
                    
                        #Load the weights
                        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                        ptr = ptr + num_biases
                    
                        #reshape the loaded weights according to the dims of the model weights
                        conv_biases = conv_biases.view_as(conv.bias.data)
                    
                        #Finally copy the data
                        conv.bias.data.copy_(conv_biases)
                        
                    #Let us load the weights for the Convolutional layers
                    num_weights = conv.weight.numel()
                    
                    noise = (torch.rand((num_weights), dtype=torch.float32) - 0.5) / 70
                    
                    #Do the same as above for weights
                    conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights]) + noise
                    ptr = ptr + num_weights
                    
                    conv_weights = conv_weights.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_weights)

    
    
    
    
    
#def get_test_input():
#    img = cv2.imread("dog-cycle-car.png")
#    img = cv2.resize(img, (416, 416))
#    img_ = img[:, :, ::-1].transpose((2, 0, 1)) # BGR -> RGB and HxWxC -> CxHxW
#    img_ = img_[np.newaxis, :, :, :]/255.0
#    img_ = torch.from_numpy(img_).float()
#    img_ = Variable(img_)
#    return img_, img


#if __name__ == "__main__":
    #cfg = parse_cfg("cfg/yolov3.cfg")
    #print(create_modules(cfg))
    
#    model = Yolo("cfg/yolov3.cfg")    
#    model.load_weights("yolov3.weights")
#    inp, out_img = get_test_input()
#    print("Cuda enabled ? ", torch.cuda.is_available())
#    pred = model(inp, torch.cuda.is_available())
#    print(pred)
#    
#    print(pred.shape)
#    filtered_res = filter_results(pred, 0.4, 80)
#    
#    boxes = filtered_res.cpu().numpy()
#    for b in boxes:
#        pt1 = (int(b[1] - b[3]/2), int(b[2] - b[4]/2))
#        pt2 = (int(b[1] + b[3]/2), int(b[2] + b[4]/2))
#        cv2.rectangle(out_img, pt1, pt2, (0, 255, 0))
#    cv2.imwrite("out.png", out_img)
        
