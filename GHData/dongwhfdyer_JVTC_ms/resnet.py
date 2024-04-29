import math
import re

from ms_converter import generate_param_mapping_ms
import numpy as np
from scipy.stats import truncnorm
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore import save_checkpoint, context, load_checkpoint, load_param_into_net
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
from config.resnet_config import config

# from t_resnet import ResNet as t_ResNet, tlayer1
# import torch
# import torch.nn as tnn


# import pydevd_pycharm
# pydevd_pycharm.settrace('120.26.167.153', port=8023, stdoutToServer=True, stderrToServer=True)

def conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0
    scale /= max(1., fan_in)
    stddev = (scale ** 0.5) / .87962566103423978
    if config.net_name == "resnet152":
        stddev = (scale ** 0.5)
    mu, sigma = 0, stddev
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size * kernel_size)
    weight = np.reshape(weight, (out_channel, in_channel, kernel_size, kernel_size))
    return Tensor(weight, dtype=mstype.float32)


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def calculate_gain(nonlinearity, param=None):
    """calculate_gain"""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    res = 0
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        res = 1
    elif nonlinearity == 'tanh':
        res = 5.0 / 3
    elif nonlinearity == 'relu':
        res = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            neg_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            neg_slope = param
        else:
            raise ValueError("neg_slope {} not a valid number".format(param))
        res = math.sqrt(2.0 / (1 + neg_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
    return res


def _calculate_fan_in_and_fan_out(tensor):
    """_calculate_fan_in_and_fan_out"""
    dimensions = len(tensor)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = tensor[1]
        fan_out = tensor[0]
    else:
        num_input_fmaps = tensor[1]
        num_output_fmaps = tensor[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = tensor[2] * tensor[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Unsupported mode {}, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_normal(inputs_shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return np.random.normal(0, std, size=inputs_shape).astype(np.float32)


def kaiming_uniform(inputs_shape, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(-bound, bound, size=inputs_shape).astype(np.float32)


def _conv3x3(in_channel, out_channel, stride=1, ):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))

    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=0, pad_mode='same', weight_init=weight)


def _conv1x1(in_channel, out_channel, stride=1, ):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))

    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                     padding=0, pad_mode='same', weight_init=weight)


def _conv7x7(in_channel, out_channel, stride=1, ):
    weight_shape = (out_channel, in_channel, 7, 7)
    weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))

    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _bn(channel, ):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9, gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _bn_last(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _fc(in_channel, out_channel, ):
    weight_shape = (out_channel, in_channel)
    weight = Tensor(kaiming_uniform(weight_shape, a=math.sqrt(5)))

    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, has_bias=False, pad_mode="pad")
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()  # kuhn edited  there is no inplace parameter in ReLU
        self.downsample = downsample

    ##########nhuk#################################### the original one
    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out
    ##########nhuk####################################


# class tBottleneck(tnn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(tBottleneck, self).__init__()
#         self.conv1 = tnn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = tnn.BatchNorm2d(planes)
#         self.conv2 = tnn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                 padding=1, bias=False)
#         self.bn2 = tnn.BatchNorm2d(planes)
#         self.conv3 = tnn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = tnn.BatchNorm2d(planes * 4)
#         self.relu = tnn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     ##########nhuk#################################### original one
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#     ##########nhuk####################################


class MaxPool2d(nn.Cell):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.max_pool = P.MaxPool(kernel_size, stride)
        self.use_pad = padding != 0
        if isinstance(padding, tuple):
            assert len(padding) == 2
            paddings = ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
        elif isinstance(padding, int):
            paddings = ((0, 0),) * 2 + ((padding, padding),) * 2
        else:
            raise ValueError('padding should be a tuple include 2 numbers or a int number')
        self.pad = P.Pad(paddings)

    def construct(self, x):
        if self.use_pad:
            x = self.pad(x)
        return self.max_pool(x)


class ResNet(nn.Cell):
    def __init__(self, block, layers, num_classes, train=True):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.istrain = train
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, has_bias=False, pad_mode="pad")
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool2d = nn.AvgPool2d((16, 8))

        self.num_features = 512
        self.feat = nn.Dense(512 * block.expansion, self.num_features)  # kuhn edited. I assueme that the weights would be replaced by ckpts,so I didn't set any speical init weights.
        self.feat_bn = nn.BatchNorm1d(self.num_features, momentum=0.9)  # todo: just add momentum parmeter to the BN.
        self.classifier = nn.Dense(self.num_features, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):  # layer1: planes: 64, blocks: 3, stride: 1
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # layer1: self.inplanes: 64, planes: 64, block.expansion: 4
            downsample = nn.SequentialCell([
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(planes * block.expansion)])
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.SequentialCell(*layers)

    def construct(self, x):
        # kuhn edited. The data type other than cell or Primitive is not allowed in Cell.construct.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool2d(x)

        x = x.view(x.shape[0], -1)

        x = self.feat(x)
        fea = self.feat_bn(x)
        fea_norm = P.L2Normalize(axis=1, epsilon=1e-12)(fea)  # kuhn: important normalize

        x = P.ReLU()(fea)
        x = self.classifier(x)

        return x, fea_norm, fea


def load_ms_resnet50_model(net=None, checkpoint=None):
    if net is None:
        net = net or ResNet(Bottleneck, [3, 4, 6, 3], config.class_num, train=False)
    if checkpoint is None:
        checkpoint = "ms_resnet50.ckpt"
    params_dict = load_checkpoint(checkpoint)
    not_loaded_params = load_param_into_net(net, params_dict)
    print("########################################")
    if len(not_loaded_params) > 0:
        print("Not loaded params:")
        for item in not_loaded_params:
            print(item)
    else:
        print("All params loaded.")

    return net


# def t_resnet50(pretrained=None, num_classes=1000, train=True):
#     model = t_ResNet(tBottleneck, [3, 4, 6, 3], num_classes, train)
#     weight = torch.load(pretrained, map_location='cpu')
#     static = model.state_dict()
#
#     base_param = []
#     for name, param in weight.items():
#         if name not in static:
#             continue
#         if isinstance(param, tnn.Parameter):
#             param = param.data
#         static[name].copy_(param)
#         base_param.append(name)
#
#     params = []
#     params_dict = dict(model.named_parameters())
#     for key, v in params_dict.items():
#         if key in base_param:
#             params += [{'params': v, 'lr_mult': 1}]
#         else:
#             # new parameter have larger learning rate
#             params += [{'params': v, 'lr_mult': 10}]
#
#     return model, params


def load_torch_model():
    # net, _ = t_resnet50(pretrained="checkpoint/resnet50_market2duke_epoch00100.pth", num_classes=751, train=False) # todo: param1
    # net, _ = t_resnet50(pretrained="checkpoint/resnet50_duke2market_epoch00100.pth", num_classes=702, train=False)
    return net


def tensor_diff_and_save_txt(m_tensor_out, t_tensor_out, comment: str = None):
    if comment is not None:
        print("######################################## tensor difference: " + comment)
    else:
        print("########################################")
    # print(np.allclose(m_tensor_out.asnumpy(), t_tensor_out.detach().numpy()) ,end=' ')
    print(np.allclose(m_tensor_out.asnumpy(), t_tensor_out.detach().numpy(), atol=1e-4), "under Precison:1e-4")
    print(np.allclose(m_tensor_out.asnumpy(), t_tensor_out.detach().numpy(), atol=1e-3), "under Precison:1e-3")
    print(np.allclose(m_tensor_out.asnumpy(), t_tensor_out.detach().numpy(), atol=1e-2), "under Precison:1e-2")
    print(np.allclose(m_tensor_out.asnumpy(), t_tensor_out.detach().numpy(), atol=1e-1), "under Precison:1e-1")
    with open("m_tensor_%s.txt" % (comment), "w") as f:
        f.write(str(m_tensor_out.asnumpy()))
    with open("t_tensor_%s.txt" % (comment), "w") as f:
        f.write(str(t_tensor_out.detach().numpy()))


def blockTest_torchVSms():
    ##########nhuk#################################### param setting


    mblock_net = ResNet(Bottleneck, [3, 4, 6, 3], 751, train=False) # todo: param2
    # mblock_net = ResNet(Bottleneck, [3, 4, 6, 3], 702, train=False)
    tblock_net = load_torch_model()


    test_input = np.random.randn(6, 3, 256, 128).astype(np.float32)  # for test resnet as a whole

    txt_save_name = "m2d" # todo: param3
    ##########nhuk####################################
    m_txt = "m_" + txt_save_name + ".txt"
    t_txt = "t_" + txt_save_name + ".txt"
    ms_ckpt = "ms_" + txt_save_name + ".ckpt"

    mblock_net.set_train(False)
    tblock_net.eval()
    generate_param_mapping_ms(mblock_net, tblock_net, m_txt, t_txt, ms_ckpt)

    mblock_net = load_ms_resnet50_model(mblock_net, ms_ckpt)
    # save_one_param_weights_numpy(mblock_net, tblock_net)

    m_tensor_in = Tensor(test_input)
    t_tensor_in = torch.from_numpy(test_input)

    #########nhuk#################################### single output
    t_tensor_out = tblock_net(t_tensor_in)[0]
    m_tensor_out = mblock_net(m_tensor_in)[0]
    #########nhuk####################################

    # ##########nhuk#################################### multi output
    # t_tensor_out, t_intermediate = tblock_net(t_tensor_in)
    # m_tensor_out, m_intermediate = mblock_net(m_tensor_in)
    # for ind, key in enumerate(t_intermediate):
    #     tensor_diff_and_save_txt(m_intermediate[ind], t_intermediate[key], key)
    # ##########nhuk####################################

    tensor_diff_and_save_txt(m_tensor_out, t_tensor_out, "final")


def save_one_param_weights_numpy(mblock_net, tblock_net):
    with open("t_bn2.txt", "w") as f:
        f.write("running mean\n")
        f.write(str(tblock_net.state_dict()['bn2.running_mean'].numpy()))
        f.write("running var\n")
        f.write(str(tblock_net.state_dict()['bn2.running_var'].numpy()))
        f.write("weight\n")
        f.write(str(tblock_net.state_dict()['bn2.weight'].numpy()))
        f.write("bias\n")
        f.write(str(tblock_net.state_dict()['bn2.bias'].numpy()))
    with open("m_bn2.txt", "w") as f:
        f.write("moving mean\n")
        f.write(str(mblock_net.parameters_dict()['bn2.moving_mean'].asnumpy()))
        f.write("moving var\n")
        f.write(str(mblock_net.parameters_dict()['bn2.moving_variance'].asnumpy()))
        f.write("weight\n")
        f.write(str(mblock_net.parameters_dict()['bn2.gamma'].asnumpy()))
        f.write("bias\n")
        f.write(str(mblock_net.parameters_dict()['bn2.beta'].asnumpy()))


if __name__ == '__main__':
    blockTest_torchVSms()
