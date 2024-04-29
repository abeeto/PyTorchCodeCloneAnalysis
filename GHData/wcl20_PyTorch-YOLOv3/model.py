import torch
import torch.nn as nn
import numpy as np

###############################################################################
# Custom Layers
###############################################################################

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors, classes, dimension):
        super(DetectionLayer, self).__init__()
        # Anchors
        self.anchors = anchors
        # Number of classes
        self.classes = classes
        # Original input image dimension
        self.dimension = dimension

    def forward(self, x):
        # Batch size N
        N = x.size(0)
        # Filter size F
        F = x.size(3)
        # Number of anchors A
        A = len(self.anchors)
        # Number of classes C
        C = self.classes
        # Reshape x from Nx(Ax(5+C))xFxF to NxFxFxAx(5+C)
        x = x.view(N, A, 5 + C, F, F)\
             .permute(0, 3, 4, 1, 2)\
             .contiguous()
        # Sigmoid center tx and center ty
        x[..., 0] = torch.sigmoid(x[..., 0])
        x[..., 1] = torch.sigmoid(x[..., 1])
        # Add offsets
        cx = torch.arange(F).repeat(F, 1).view([1, F, F, 1])
        cy = torch.arange(F).repeat(F, 1).t().view([1, F, F, 1])
        x[..., 0] += cx
        x[..., 1] += cy
        # Get filter scale
        scale = self.dimension / F
        # Scale anchors
        anchors = torch.FloatTensor([(w / scale, h / scale) for (w, h) in self.anchors])
        # Compute width
        x[..., 2] = torch.exp(x[..., 2]) * anchors[:,0].view(1, 1, 1, A)
        # Compute height
        x[..., 3] = torch.exp(x[..., 3]) * anchors[:,1].view(1, 1, 1, A)
        # Scale prediction box
        x[..., :4] *= scale
        # Sigmoid object confidence
        x[..., 4] = torch.sigmoid(x[..., 4])
        # Sigmoid class predictions
        x[..., 5:] = torch.sigmoid(x[..., 5:])
        # Output
        output = x.view(N, F * F * A, 5 + C)
        return output

###############################################################################
# Helper Functions
###############################################################################

def create_blocks(filename):
    # Parse config file
    file = open(filename, "r")
    lines = file.read().split('\n')
    # Remove empty lines and comments
    lines = [x for x in lines if x and x[0] != '#']
    # Remove white spaces
    lines = [x.rstrip().lstrip() for x in lines]
    # Block definitions
    block_list = []
    for line in lines:
        # Start of new block
        if line[0] == "[":
            block_list.append({})
            block_list[-1]["type"] = line[1:-1].rstrip()
        # Define block attributes
        else:
            key, value = line.split("=")
            block_list[-1][key.rstrip()] = value.lstrip()
    return block_list[0], block_list[1:]

def create_modules(info, block_list):
    # Output filter size list
    output_list = [3]
    # Module list
    module_list = nn.ModuleList()
    for i, block in enumerate(block_list):

        # Create module
        module = nn.Sequential()

        # Convolutional Layer
        if block["type"] == "convolutional":
            # Attributes
            input = output_list[-1]
            output = int(block["filters"])
            size = int(block["size"])
            stride = int(block["stride"])
            padding = (size - 1) // 2
            batch = int(block["batch_normalize"]) if "batch_normalize" in block else 0
            activation = block["activation"]
            # Downsampling
            module.add_module(f"conv_{i}", nn.Conv2d(input, output, kernel_size=size, stride=stride, padding=padding, bias=not batch))
            # Batch Normalization
            if batch:
                module.add_module(f"batch_norm_{i}", nn.BatchNorm2d(output))
            # Leaky ReLU
            if activation == "leaky":
                module.add_module(f"leaky_{i}", nn.LeakyReLU(0.1, inplace=True))

        # Upsample Layer
        elif block["type"] == "upsample":
            # Attributes
            stride = int(block["stride"])
            # Upsampling
            module.add_module(f"upsample_{i}", nn.Upsample(scale_factor=stride, mode="nearest"))

        # Route Layer
        elif block["type"] == "route":
            # Get all input layer of route (Either 1 or 2 inputs)
            inputs = [int(layer) for layer in block["layers"].split(",")]
            output = sum([output_list[1:][i] for i in inputs])
            # Dummy layer to handle route connection
            module.add_module(f"route_{i}", EmptyLayer())

        # Shortcut Layer
        elif block["type"] == "shortcut":
            # Filter size should be equal before and after skip
            assert output_list[-1] == output_list[int(block["from"])]
            # Dummy layer to handle skip connection
            module.add_module(f"shortcut_{i}", EmptyLayer())

        # YOLO Layer
        elif block["type"] == "yolo":
            # Get anchor indexes
            indices = [int(i) for i in block["mask"].split(",")]
            anchors = [int(i) for i in block["anchors"].split(",")]
            anchors = [(anchors[2 * i], anchors[2 * i + 1]) for i in indices]
            classes = int(block["classes"])
            dimension = int(info["height"])
            module.add_module(f"yolo_{i}", DetectionLayer(anchors, classes, dimension))

        else:
            print("Unknown block type")

        # Add module to module list
        module_list.append(module)
        # Update output filter sizes
        output_list.append(output)

    return module_list

###############################################################################
# YOLO v3
###############################################################################

class YOLO(nn.Module):
    def __init__(self, filename):
        super(YOLO, self).__init__()
        self.info, self.block_list = create_blocks(filename)
        self.module_list = create_modules(self.info, self.block_list)

    def forward(self, x):
        # Layer output list
        output_list = []
        # Final output
        output = []
        for i, (block, module) in enumerate(zip(self.block_list, self.module_list)):

            # Convolutional / Upsample Layer
            if block["type"] in ["convolutional", "upsample"]:
                # Pass into Sequential
                x = module(x)

            # Route Layer
            elif block["type"] == "route":
                # Get all input layer of route (Either 1 or 2 inputs)
                inputs = [int(layer) for layer in block["layers"].split(",")]
                # Concatenate inputs
                x = torch.cat([output_list[i] for i in inputs], 1)

            # Shortcut Layer
            elif block["type"] == "shortcut":
                # Add previous output with skip connection
                x = output_list[-1] + output_list[int(block["from"])]

            # YOLO Layer
            elif block["type"] == "yolo":
                x = module(x)
                output.append(x)

            else:
                print("Unknown block type")

            # Add output to output list
            output_list.append(x)

        return torch.cat(output, 1)

    def load_weights(self, filename):
        file = open(filename, "rb")
        # The first 5 items is header
        header = np.fromfile(file, dtype=np.int32, count=5)
        # The remaining values are the weights
        weights = np.fromfile(file, dtype=np.float32)
        # Weight pointer
        ptr = 0
        for i, (block, module) in enumerate(zip(self.block_list, self.module_list)):

            # Only convolutional Layers have weight
            if block["type"] == "convolutional":
                convolution_layer = module[0]
                # If the block uses batch normalisation
                if "batch_normalize" in block:

                    batchnorm_layer = module[1]
                    # Get number of elements
                    el = batchnorm_layer.bias.numel()
                    # Load biases
                    batch_norm_bias = torch.from_numpy(weights[ptr : ptr + el])
                    ptr += el
                    # Load weights
                    batch_norm_weight = torch.from_numpy(weights[ptr : ptr + el])
                    ptr += el
                    # Load running mean
                    batch_norm_mean = torch.from_numpy(weights[ptr : ptr + el])
                    ptr += el
                    # Loading running variance
                    batch_norm_var = torch.from_numpy(weights[ptr : ptr + el])
                    ptr += el

                    # Reshape
                    batch_norm_bias = batch_norm_bias.view_as(batchnorm_layer.bias)
                    batch_norm_weight = batch_norm_weight.view_as(batchnorm_layer.weight)
                    batch_norm_mean = batch_norm_mean.view_as(batchnorm_layer.running_mean)
                    batch_norm_var = batch_norm_var.view_as(batchnorm_layer.running_var)

                    # Copy
                    batchnorm_layer.bias.data.copy_(batch_norm_bias)
                    batchnorm_layer.weight.data.copy_(batch_norm_weight)
                    batchnorm_layer.running_mean.data.copy_(batch_norm_mean)
                    batchnorm_layer.running_var.data.copy_(batch_norm_var)

                else:

                    # Otherwise, load convolutional bias
                    el = convolution_layer.bias.numel()
                    # Load biases
                    convolutional_bias = torch.from_numpy(weights[ptr : ptr + el])
                    ptr += el
                    # Reshape
                    convolutional_bias = convolutional_bias.view_as(convolution_layer.bias)
                    # Copy
                    convolution_layer.bias.data.copy_(convolutional_bias)

                # Load convolutional weights
                el = convolution_layer.weight.numel()
                # Load weights
                convolutional_weight = torch.from_numpy(weights[ptr : ptr + el])
                ptr += el
                # Reshape
                convolutional_weight = convolutional_weight.view_as(convolution_layer.weight)
                # Copy
                convolution_layer.weight.data.copy_(convolutional_weight)
