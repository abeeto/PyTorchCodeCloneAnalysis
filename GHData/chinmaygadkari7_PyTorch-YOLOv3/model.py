import os
import torch
from torch import nn
import numpy as np

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x

class Upsample(nn.Module):
    """Upsample layer"""
    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

class YOLO(nn.Module):
    def __init__(self, anchors, num_classes):
        super(YOLO, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.num_attribs = num_classes + 5 # [x,y,w,h,p_obj,[p_classes...]]

    def forward(self, p, image_size):
        batch_size, ny, nx = p.size(0), p.size(2), p.size(3)
        p = p.view(batch_size, self.num_anchors, self.num_attribs, ny, nx)
        p = p.permute(0, 1, 3, 4, 2).contiguous()

        if self.training:
            return p

        strides = image_size // max(nx, ny)
        ys, xs = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        grid_xy = torch.stack((xs, ys), 2).float().view((1, 1, nx, ny, 2))
        anchors = self.anchors // strides
        anchors = anchors.view(1, self.num_anchors, 1, 1, 2)
        p[..., 0:2] = torch.sigmoid(p[...,0:2]) + grid_xy
        p[..., 2:4] = torch.exp(p[...,2:4]) * anchors
        p[..., 4: ] = torch.sigmoid(p[...,4:])
        p[...,  :4] = p[...,  :4] * strides

        return p.view(batch_size, -1, self.num_attribs)

class Darknet(nn.Module):
    """DarkNet Model"""
    def __init__(self, configuration):
        super(Darknet, self).__init__()
        self.configuration = configuration
        self.params, self.modules_list = create_modules(configuration)

    def forward(self, x):
        image_size = max(x.shape[-2:])
        layer_outputs = []
        output = []

        for i, (config, module) in enumerate(zip(self.configuration[1:], self.modules_list)):
            module_type = config['type']

            if module_type in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)

            elif module_type == 'route':
                layer_ids = [
                    int(l) for l in config['layers'].split(',')
                    ]

                if len(layer_ids) > 1:
                    x = torch.cat([
                        layer_outputs[idx] for idx in layer_ids
                        ], 1)
                else:
                    x = layer_outputs[layer_ids[0]]

            elif module_type == 'shortcut':
                layer_id = int(config['from'])
                x = layer_outputs[-1] + layer_outputs[layer_id]

            elif module_type == 'yolo':
                x = module[0](x, image_size)
                output.append(x)

            layer_outputs.append(x)

        if self.training:
            return output

        return torch.cat(output, 1)

    def load_weights(self, weights_file):
        with open(weights_file, 'rb') as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)

        ptr = 0
        for idx, (config, module) in enumerate(zip(self.configuration[1:], self.modules_list)):
            if config['type'] == 'convolutional':
                conv = module[0]
                batch_normalize = bool(config.get('batch_normalize', 0))
                if batch_normalize:
                    bn = module[1]

                    num_bn_biases = bn.bias.numel()
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases]).view_as(bn.bias.data)
                    bn.bias.data.data.copy_(bn_biases)
                    ptr += num_bn_biases

                    num_bn_weights = bn.weight.numel()
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_weights]).view_as(bn.weight.data)
                    bn.weight.data.data.copy_(bn_weights)
                    ptr  += num_bn_weights

                    num_bn_running_mean = bn.running_mean.numel()
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_running_mean]).view_as(bn.running_mean.data)
                    bn.running_mean.data.copy_(bn_running_mean)
                    ptr  += num_bn_running_mean

                    num_bn_running_var = bn.running_var.numel()
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_running_var]).view_as(bn.running_var.data)
                    bn.running_var.data.copy_(bn_running_var)
                    ptr  += num_bn_running_var

                else:
                    num_bias = conv.bias.numel()
                    conv_bias = torch.from_numpy(weights[ptr:ptr+num_bias]).view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_bias)
                    ptr+= num_bias

                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights]).view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
                ptr += num_weights


def create_modules(configuration):
    """
    Create list of modules from list of module parameters

    Parameters:
        module_definations: List of dict
    """

    params = configuration[0] # net module
    output_filters = [
        int(params['channels'])
        ]

    modules_list = nn.ModuleList()
    for index, box in enumerate(configuration[1:]):
        modules = nn.Sequential()
        module_type = box['type']

        if  module_type == 'convolutional':
            kernel_size = int(box['size'])
            stride = int(box['stride'])
            filters = int(box['filters'])
            in_channels = output_filters[-1]
            use_padding = bool(box['pad'])
            if use_padding:
                padding = (kernel_size - 1) // 2
            else:
                padding = 0

            activation = box['activation']
            batch_normalize = bool(int(box.get('batch_normalize', 0)))
            bias = True if not batch_normalize else False

            modules.add_module('conv_{}'.format(index),
                nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=bias
                    )
                )

            if batch_normalize:
                modules.add_module('batch_norm_{}'.format(index),
                    nn.BatchNorm2d(filters)
                )

            if activation == 'leaky':
                modules.add_module('leaky_{}'.format(index),
                    nn.LeakyReLU(0.1, inplace=True)
                )

        elif module_type == 'maxpool':
            kernel_size = int(box['size'])
            stride = int(box['stride'])
            padding = kernel_size - 1 // 2
            if stride == 1:
                modules.add_module('padding_{}'.format(index),
                    nn.ZeroPad2d((0,1,0,1))
                    )

            maxpool = nn.MaxPool2d(
                kernel_size=kernel_size,
                stide=stride,
                padding=padding
                )

            modules.add_module('maxpool_{}'.format(index), maxpool)

        elif module_type == 'upsample':
            stride = int(box['stride'])
            modules.add_module('upsample_{}'.format(index),
                    Upsample(scale_factor=stride)
                )

        elif module_type == 'route':
            filters = 0
            layers = [int(l) for l in box['layers'].split(',')]
            for layer in layers:
                if layer > 0 and (layer -index) < 0:
                    filters += output_filters[layer]

                else:
                    filters += output_filters[layer]

            modules.add_module('route_{}'.format(index), EmptyLayer())

        elif module_type == 'shortcut':
            from_module = int(box['from'])
            filters = output_filters[from_module]
            modules.add_module('shortcut_{}'.format(index), EmptyLayer())

        elif module_type == 'yolo':
            mask = [int(mask) for mask in box['mask'].split(',')]
            anchors = [int(anchor) for anchor in box['anchors'].split(',')]
            anchors = [ (anchors[i], anchors[i + 1])
                for i in range(0, len(anchors), 2)
            ]
            anchors = [anchors[i] for i in mask]
            num_classes = int(box['classes'])
            image_size = params['height']
            modules.add_module('yolo_{}'.format(index),
                YOLO(anchors, num_classes)
            )

        modules_list.append(modules)
        output_filters.append(filters)

    return params, modules_list
