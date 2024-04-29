#pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring, too-many-locals, too-many-statements, too-many-branches
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
import numpy as np
import cv2
import onnxruntime as rt
from util import *

# from torchsummary import summary
# from winmltools import convert_coreml

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (224, 224))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_


def parse_cfg(cfgfile):
    file = open(cfgfile, "r")
    lines = file.read().split("\n")
    lines = [l for l in lines if len(l) > 0]
    lines = [l for l in lines if l[0] != "#"]
    lines = [l.rstrip().lstrip() for l in lines]
    file.close()

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1: -1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks


def create_module(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for idx, blk in enumerate(blocks[1:]):
        module = nn.Sequential()

        if blk["type"] == "convolutional":
            activation = blk["activation"]
            try:
                batch_normalize = int(blk["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(blk["filters"])
            padding = int(blk["pad"])
            kernel_size = int(blk["size"])
            stride = int(blk["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size,
                             stride=stride, padding=pad, bias=bias)
            module.add_module("conv_{}".format(idx), conv)

            if batch_normalize:
                batch_norm = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{}".format(idx), batch_norm)
            if activation == "leaky":
                activation_ = nn.LeakyReLU(0.1, True)
                module.add_module("Leaky_{}".format(idx), activation_)
        elif blk["type"] == "upsample":
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample_{}".format(idx), upsample)

        elif blk["type"] == "route":
            blk["layers"] = blk["layers"].split(",")
            start = int(blk["layers"][0])

            try:
                end = int(blk["layers"][1])
            except:
                end = 0

            if start > 0:
                start = start - idx
            if end > 0:
                end = end - idx
            route = EmptyLayer()
            module.add_module("route_{0}".format(route), route)
            if end < 0:
                filters = output_filters[idx + start] + output_filters[idx + end]
            else:
                filters = output_filters[idx + start]

        elif blk["type"] == "shortcut":
            _ = int(blk["from"])
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(idx), shortcut)

        elif blk["type"] == "maxpool":
            stride = int(blk["stride"])
            size = int(blk["size"])
            if size != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)

            module.add_module("maxpool_{}".format(idx), maxpool)

        elif blk["type"] == "yolo":
            mask = blk["mask"].split(",")
            mask = [int(val) for val in mask]

            anchors = blk["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(idx), detection)
        else:
            pass

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return net_info, module_list


class EmptyLayer(nn.Module):
    pass


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

    def forward(self, x, inp_dim, num_classes, confidence): #pylint: disable=arguments-differ
        x = x.data
        prediction = x
        prediction = predict_transform(prediction, inp_dim, self.anchors, num_classes, confidence)
        return prediction


class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x): #pylint: disable=arguments-differ
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x


class DarkNet(nn.Module):
    def __init__(self, cfgfile, weights_file=None):
        super(DarkNet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_module(self.blocks)
        if weights_file:
            self.load_weights(weights_file)

    def get_blocks(self):
        return self.blocks

    def get_moduel_list(self):
        return self.module_list

    def forward(self, x, CUDA=False):
        detections = []
        modules = self.blocks[1:]
        outputs = {}
        write = 0

        for i, module in enumerate(modules):
            module_type = (module["type"])
            if module_type in ("convolutional", "upsample", "maxpool"):
                x = self.module_list[i](x)
                outputs[i] = x

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)
                outputs[i] = x

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]
                outputs[i] = x

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(module["classes"])

                # Output the result
                #x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)

                #if type(x) == int:
                #    continue

                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            #outputs[i] = outputs[i - 1]
            outputs[i] = x
        return detections

    def load_weights(self, weightfile):
        fileptr = open(weightfile, "rb")
        header = np.fromfile(fileptr, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fileptr, dtype=np.float32)
        ptr = 0

        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]
                if batch_normalize:
                    batch_norm = model[1]
                    num_bn_biases = batch_norm.bias.numel()

                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_biases = bn_biases.view_as(batch_norm.bias.data)
                    bn_weights = bn_weights.view_as(batch_norm.weight.data)
                    bn_running_mean = bn_running_mean.view_as(batch_norm.running_mean)
                    bn_running_var = bn_running_var.view_as(batch_norm.running_var)

                    batch_norm.bias.data.copy_(bn_biases)
                    batch_norm.weight.data.copy_(bn_weights)
                    batch_norm.running_mean.copy_(bn_running_mean)
                    batch_norm.running_var.copy_(bn_running_var)

                else:
                    num_biases = conv.bias.numel()

                    conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
                    ptr += num_biases

                    conv_biases = conv_biases.view_as(conv.bias.data)

                    conv.bias.data.copy_(conv_biases)
                num_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


def main():
    #net_scripted = torch.jit.script(DarkNet("cfg/yolov3.cfg", "yolov3.weights"))
    #dummy_input = torch.ones((1, 3, 224, 224))
    #output = net_scripted(dummy_input)

    #torch.onnx.export(net_scripted,
    #                  (dummy_input),
    #                  'model.onnx',
    #                  verbose=True,
    #                  input_names=['input_data'],
    #                  example_outputs=output)

    net = DarkNet("cfg/yolov3.cfg")
    net.load_weights("yolov3.weights")

    for param in net.parameters():
        param.requires_grad = False

    in_val = get_test_input()
    output = net(in_val, False)
    torch.onnx.export(net,
                      torch.ones((1, 3, 224, 224)),
                      "model.onnx",
                      export_params=True,
                      verbose=True,
                      input_names=['input_data'],
                      example_outputs=output)

"""
def main():
    net = DarkNet("cfg/yolov3.cfg")
    net.load_weights("yolov3.weights")

    for param in net.parameters():
        param.requires_grad = False

    in_val = get_test_input()
    output = net(in_val, False)
    torch.onnx.export(net,
                      torch.ones((1, 3, 224, 224)),
                      "model.onnx",
                      export_params=True,
                      verbose=True,
                      input_names=['input_data'],
                      example_outputs=output,
                      opset_version=11)

    #sess = rt.InferenceSession("model.onnx")
    #print(sess)

    #print(summary(net, (3, 224, 224)))
"""
if __name__ == '__main__':
    main()