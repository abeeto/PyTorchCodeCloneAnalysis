# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import sys
import torch
import torch.nn as nn
import math
from Args.args import ARGS
import numpy as np

# 取每个layer的最后一层的hwc
HW_Layer = {
    'layer1': [56, 56, 64],
    'layer2': [28, 28, 128],
    'layer3': [14, 14, 256],
    'layer4': [7, 7, 512],
} if (ARGS.model == 'GCN' and (not ARGS.GCN_is_maxpool)) else {
    'layer1': [28, 28, 64],
    'layer2': [14, 14, 128],
    'layer3': [7, 7, 256],
    'layer4': [4, 4, 512],
}


class ChannelEnhance_1(nn.Module):
    # 第一种尝试:
    # 用channel中的(:fold)+(fold:2 * fold)|(fold:2 * fold)+(2 * fold:3 * fold),来增强(:fold)|(fold: 2 * fold)
    def __init__(self, net, n_div=8):
        super(ChannelEnhance_1, self).__init__()
        self.net = net
        self.n_div = n_div

    def forward(self, x):
        x = self.enhance(x, n_div=self.n_div)
        return self.net(x)

    def enhance(self, x, n_div=8):
        bs, c, h, w = x.size()
        if ARGS.model == 'GCN':
            c = c // 4
            x = x.reshape(bs, c, 4, h, w)

        assert c % n_div == 0, 'enhance()!!!'
        fold = c // n_div

        # TSM
        out = torch.zeros_like(x)
        out[:, :fold, :] = x[:, :fold, :] + x[:, fold:2 * fold, :]

        if n_div == 2:
            out[:, fold:2 * fold, :] = x[:, fold:2 * fold, :]
        else:
            out[:, fold: 2 * fold, :] = x[:, fold: 2 * fold, :] + x[:, 2 * fold:3 * fold, :]
            out[:, 2 * fold:, :] = x[:, 2 * fold:, :]

        out = out.reshape(bs, 4 * c, h, w) if len(out.shape) == 5 else out

        return out


class ChannelEnhance_2(nn.Module):
    # 第二种尝试:
    # 在特征图(c*hw)中用卷积核为(c//8-1)的DepthWise Conv对channel进行卷积,实现在c//8 channels的范围下进行enhance;
    def __init__(self, net, h, w, input_channels, n_div, block_position, init_strategy, kernelsize):
        super(ChannelEnhance_2, self).__init__()
        self.net = net
        M = 4 if ARGS.model == 'GCN' else 1
        # Attention!!注意这里的fold是指channel shift的fold(包含左移/右移),因此在channel enhance中需要将取两份fold;
        self.fold = M * h * w // n_div
        self.conv = nn.Conv1d(in_channels=M * h * w, out_channels=M * h * w, kernel_size=kernelsize,
                              padding=math.floor(kernelsize / 2), groups=M * h * w, bias=False)

        self.conv, self.i = init_conv(self.conv, n_div, M * h * w, block_position, self.fold,
                                      init_strategy, kernelsize, channel_enhance=2)
        save_fig_for_all_conv_config(False, init_conv, M * h * w)
        show_conv_weight(False, self.conv)

    def forward(self, x):
        x = enhance(x, self.conv, self.i, self.fold)
        return self.net(x)


class ChannelEnhance_3(nn.Module):
    # 第三种尝试:
    # cross channel;
    def __init__(self, net, h, w, input_channels, n_div=8, block_position='start'):
        super(ChannelEnhance_3, self).__init__()
        self.net = net
        M = 4 if ARGS.model == 'GCN' else 1
        self.fold = M * h * w // n_div
        self.conv = nn.Conv1d(in_channels=M * h * w, out_channels=M * h * w, kernel_size=3,
                              padding=1, groups=M * h * w, bias=False)

        self.conv, self.i = init_conv(self.conv, n_div, M * h * w, block_position, self.fold)
        save_fig_for_all_conv_config(False, init_conv, M * h * w)
        show_conv_weight(False, self.conv)

    def forward(self, x):
        x = enhance(x, self.conv, self.i, self.fold)
        return self.net(x)


class ChannelEnhance_4(nn.Module):
    # 第四种尝试:
    # cross channel n enhance channel;
    # TODO:当同时采用shift和enhance时,我假设二者同时插入进同一个layer,但block_position和n_div则不同;
    def __init__(self, net, h, w, input_channels, n_div_s, block_position_s, n_div_e, block_position_e, init_strategy_e,
                 kernelsize_e):
        super(ChannelEnhance_4, self).__init__()
        self.net = net
        M = 4 if ARGS.model == 'GCN' else 1
        self.fold_s, self.fold_e = M * h * w // n_div_s, M * h * w // n_div_e

        # kernel for shift
        self.conv_s = nn.Conv1d(in_channels=M * h * w, out_channels=M * h * w, kernel_size=3,
                                padding=1, groups=M * h * w, bias=False)
        self.conv_s, self.i_s = init_conv(self.conv_s, n_div_s, M * h * w, block_position_s, self.fold_s)

        # kernel for enhance
        self.conv_e = nn.Conv1d(in_channels=M * h * w, out_channels=M * h * w, kernel_size=kernelsize_e,
                                padding=math.floor(kernelsize_e / 2), groups=M * h * w, bias=False)
        self.conv_e, self.i_e = init_conv(self.conv_e, n_div_e, M * h * w, block_position_e, self.fold_e, init_strategy_e,
                                        kernelsize_e, channel_enhance=2)

    def forward(self, x):
        x = enhance(x, [self.conv_s, self.conv_e], [self.i_s, self.i_e], [self.fold_s, self.fold_e])
        return self.net(x)


# def residual(x, identity, i, fold, is_residual):
#     if is_residual:
#         bs, c, h, w = identity.size()
#         identity = identity.reshape(bs, c // 4, 4 * h * w) if ARGS.model == 'GCN' else x.reshape(bs, c, h * w)
#         identity = identity.permute([0, 2, 1])
#
#         if ARGS.channel_block_position == 'stochastic':
#             x[:, i, :] = identity[:, i, :] + x[:, i, :]
#         else:
#             x[:, i * fold:(i + 2) * fold, :] = \
#                 identity[:, i * fold:(i + 2) * fold, :] + x[:, i * fold:(i + 2) * fold, :]
#
#     return x


def residual_conv(x, conv, i, fold, channel_block_position):
    identity = x
    x = conv(x)

    if channel_block_position == 'stochastic':
        x[:, i, :] = identity[:, i, :] + x[:, i, :]
    else:
        x[:, i * fold:(i + 2) * fold, :] = \
            identity[:, i * fold:(i + 2) * fold, :] + x[:, i * fold:(i + 2) * fold, :]

    return x


def fusion_SE(x, conv, i, fold):
    conv_s, conv_e = conv[0], conv[1]
    i_s, i_e = i[0], i[1]
    fold_s, fold_e = fold[0], fold[1]

    if ARGS.fusion_SE == 'A':
        x = conv_s(x)
        x = residual_conv(x, conv_e, i_e, fold_e, ARGS.channel_block_position_e) \
            if ARGS.TSM_conv_insert_e == 'residual' else conv_e(x)

    elif ARGS.fusion_SE == 'B':
        x_t = x
        x = conv_s(x)
        x_t = residual_conv(x_t, conv_e, i_e, fold_e, ARGS.channel_block_position_e) \
            if ARGS.TSM_conv_insert_e == 'residual' else conv_e(x_t)
        x = x + x_t

    elif ARGS.fusion_SE == 'C':
        x = conv_s(x)
        x_t = residual_conv(x, conv_e, i_e, fold_e, ARGS.channel_block_position_e) \
            if ARGS.TSM_conv_insert_e == 'residual' else conv_e(x)
        x = x + x_t

    else:
        assert False, 'fusion_SE()!!!'

    return x


def enhance(x, conv, i, fold):
    bs, c, h, w = x.size()
    x = x.reshape(bs, c // 4, 4 * h * w) if ARGS.model == 'GCN' else x.reshape(bs, c, h * w)
    # (bs, 4*h*w, c//4)/(bs, h*w, c)
    x = x.permute([0, 2, 1])

    if ARGS.TSM_channel_enhance in [2, 3]:
        x = residual_conv(x, conv, i, fold, ARGS.channel_block_position) if ARGS.TSM_conv_insert_e == 'residual' else conv(x)

    elif ARGS.TSM_channel_enhance in [4]:
        x = fusion_SE(x, conv, i, fold)

    else:
        assert False, 'enhance()!!!'

    # (bs, c//4, 4*h*w)/(bs, c, h*w)
    x = x.permute([0, 2, 1])
    x = x.reshape(bs, c, h, w)

    return x


def make_channel_shift(net):
    assert ARGS.base_model in ['resnet18', 'resnet34'], 'make_temporal_shift()!!!'

    # args.TSM_position: ['layer1'] or ['layer1', 'layer3']
    for position in ARGS.TSM_position:
        layer = getattr(net, position)
        layer = make_BasicBlock_shift_test(layer, HW_Layer[position], ARGS.TSM_div, ARGS.TSM_module_insert)

        setattr(net, position, layer)

    return net


# def make_BasicBlock_shift(stage, hwc, n_div, mode='inplace'):
#     # mode:inplace/residual
#     # 在每一个blocks的最后一个conv进行channel shift
#     blocks = list(stage.children())
#
#     if ARGS.TSM_channel_enhance == 1:
#         blocks[-1] = ChannelEnhance_1(blocks[-1], n_div=n_div)
#     elif ARGS.TSM_channel_enhance == 2:
#         blocks[-1] = ChannelEnhance_2(blocks[-1], hwc[0], hwc[1], hwc[2],
#                                       n_div=n_div, block_position=ARGS.channel_block_position,
#                                       init_strategy=ARGS.channel_enhance_init_strategy,
#                                       kernelsize=ARGS.channel_enhance_kernelsize)
#     elif ARGS.TSM_channel_enhance == 3:
#         blocks[-1] = ChannelEnhance_3(blocks[-1], hwc[0], hwc[1], hwc[2],
#                                       n_div=n_div, block_position=ARGS.channel_block_position)
#     else:
#         assert False, 'make_BasicBlock_shift()'
#
#     return nn.Sequential(*blocks)


def make_BasicBlock_shift_test(stage, hwc, n_div, mode='residual'):
    # mode:in-place/residual
    # 在每一个blocks的最后一个conv进行channel shift
    assert mode in ['inplace', 'residual'], 'make_BasicBlock_shift_test()!!!'

    blocks = list(stage.children())
    modified_module = blocks[-1] if mode == 'inplace' else blocks[-1].conv1
    if ARGS.TSM_channel_enhance == 1:
        modified_module = ChannelEnhance_1(modified_module, n_div=n_div)
    elif ARGS.TSM_channel_enhance == 2:
        modified_module = ChannelEnhance_2(modified_module, hwc[0], hwc[1], hwc[2],
                                           n_div=n_div, block_position=ARGS.channel_block_position,
                                           init_strategy=ARGS.channel_enhance_init_strategy,
                                           kernelsize=ARGS.channel_enhance_kernelsize)
    elif ARGS.TSM_channel_enhance == 3:
        modified_module = ChannelEnhance_3(modified_module, hwc[0], hwc[1], hwc[2],
                                           n_div=n_div, block_position=ARGS.channel_block_position)

    elif ARGS.TSM_channel_enhance == 4:
        modified_module = ChannelEnhance_4(modified_module, hwc[0], hwc[1], hwc[2],
                                           n_div_s=n_div, block_position_s=ARGS.channel_block_position,
                                           n_div_e=ARGS.TSM_div_e, block_position_e=ARGS.channel_block_position_e,
                                           init_strategy_e=ARGS.channel_enhance_init_strategy,
                                           kernelsize_e=ARGS.channel_enhance_kernelsize)
    else:
        assert False, 'make_BasicBlock_shift()'

    if mode == 'inplace':
        blocks[-1] = modified_module
    else:
        blocks[-1].conv1 = modified_module

    return nn.Sequential(*blocks)


# 在之前的模型中运用了这个Module,torch.load()需要保存此代码;与class ChannelEnhance_1相同
class ChannelShift(nn.Module):
    # 第一种尝试:
    # 用channel中的(:fold)+(fold:2 * fold)|(fold:2 * fold)+(2 * fold:3 * fold),来增强(:fold)|(fold: 2 * fold)
    def __init__(self, net, n_div=8):
        super(ChannelShift, self).__init__()
        self.net = net
        self.fold_div = n_div

    def forward(self, x):
        x = self.enhance(x, fold_div=self.fold_div)
        return self.net(x)

    @staticmethod
    def enhance(x, fold_div=8):
        bs, c, h, w = x.size()
        if ARGS.model == 'GCN':
            c = c // 4
            x = x.reshape(bs, c, 4, h, w)

        fold = c // fold_div

        # TSM
        out = torch.zeros_like(x)
        out[:, :fold, :] = x[:, :fold, :] + x[:, fold:2 * fold, :]
        out[:, fold: 2 * fold, :] = x[:, fold: 2 * fold, :] + x[:, 2 * fold:3 * fold, :]
        out[:, 2 * fold:, :] = x[:, 2 * fold:, :]
        out = out.reshape(bs, 4 * c, h, w) if len(out.shape) == 5 else out

        return out


def code_for_paper(h, w, sigma, i, x):
    # initialization
    conv = nn.Conv1d(in_channels=h * w, out_channels=h * w, kernel_size=3, padding=1, groups=h * w, bias=False)

    b_size = h * w / sigma  # block size
    conv.weight.requires_grad = False
    conv.weight.data.zero_()
    conv.weight.data[i * b_size:(i + 1) * b_size, 0, 2] = 1
    conv.weight.data[(i + 1) * b_size: (i + 2) * b_size, 0, 0] = 1
    conv.weight.data[:i * b_size, 0, 1] = 1
    conv.weight.data[(i + 2) * b_size:, 0, 1] = 1

    # forward
    bs, c, h, w = x.size()
    x = x.reshape(bs, c, h * w).permute([0, 2, 1])
    x = conv(x)
    x = x.permute([0, 2, 1]).reshape(bs, c, h, w)


def init_conv(conv, n_div, hw, block_position, fold, init_strategy=3, kernelsize=3, channel_enhance=3):
    # init strategy
    # 1.init for 1;
    # 2.default init;
    # 3.init for shift;

    if channel_enhance == 3:
        conv.weight.requires_grad = False

    # conv.weight.shape:(h * w, 1, 3)
    nn.init.constant_(conv.weight, 0)

    if block_position == 'start':
        i = 0
    elif block_position == 'middle':
        i = int(n_div // 2 - 1)
    elif block_position == 'end':
        i = n_div - 2
    elif block_position == 'stochastic':
        # 从hw中随机选取2*self.fold个数
        i = np.random.choice(np.arange(hw), 2 * fold, replace=False)
        if channel_enhance == 2 or channel_enhance == 4:
            global STOCHASTIC_I
            STOCHASTIC_I = i
    else:
        assert False, 'init_conv()!!!'

    if block_position == 'stochastic':
        conv = init_conv_stochastic(conv, i, hw, fold, init_strategy, kernelsize)
    else:
        conv = init_conv_for_others(conv, i, hw, fold, init_strategy, kernelsize)

    return conv, i


def init_conv_for_others(conv, i, hw, fold, init_strategy, kernelsize):
    middle_i = math.floor(kernelsize / 2)

    # fixed
    if i * fold != 0:
        conv.weight.data[:i * fold, 0, middle_i] = 1
    if (i + 2) * fold < hw:
        conv.weight.data[(i + 2) * fold:, 0, middle_i] = 1

    # init for 1;
    if init_strategy == 1:
        conv.weight.data[i * fold:(i + 2) * fold, 0, :] = 1

    # defult init-kaiming_uniform_
    elif init_strategy == 2:
        nn.init.kaiming_uniform_(conv.weight.data[i * fold:(i + 2) * fold, 0, :])

    # init for shift
    elif init_strategy == 3:
        conv.weight.data[i * fold:(i + 1) * fold, 0, middle_i + 1] = 1  # shift left
        conv.weight.data[(i + 1) * fold: (i + 2) * fold, 0, middle_i - 1] = 1  # shift right

    return conv


def init_conv_stochastic(conv, i, hw, fold, init_strategy, kernelsize):
    middle_i = math.floor(kernelsize / 2)

    # fixed
    conv.weight.data[[k for k in set(range(hw)) - set(i)], 0, middle_i] = 1

    # init: 1;
    if init_strategy == 1:
        conv.weight.data[i, 0, :] = 1  # shift left

    # defult init: kaiming_uniform_
    elif init_strategy == 2:
        # 发现直接进行nn.init.kaiming_uniform_(conv.weight.data[i, 0, :])行不通
        tem = conv.weight.data[i, 0, :]
        nn.init.kaiming_uniform_(tem)

        for k, t in zip(i, tem):
            conv.weight.data[k, 0, :] = t

    # init: shift
    elif init_strategy == 3:
        conv.weight.data[i[:fold], 0, middle_i + 1] = 1  # shift left
        conv.weight.data[i[fold:], 0, middle_i - 1] = 1  # shift right

    return conv


def show_conv_weight(is_show, conv):
    if is_show:
        import matplotlib.pyplot as plt
        if type(conv) == torch.Tensor:
            plt.imshow(conv.squeeze().permute(1, 0))
        else:
            plt.imshow(conv.weight.data.squeeze().permute(1, 0))

        plt.show()


def save_fig_for_all_conv_config(is_save, init_conv, Mhw):
    if is_save:
        assert ARGS.TSM_channel_enhance in [2, 3], 'save_fig_for_all_conv_config!!!'
        import matplotlib.pyplot as plt

        kernelsize_list = [3, 5, 7, 9] if ARGS.TSM_channel_enhance == 2 else [3]
        init_strategy_list = [1, 2, 3] if ARGS.TSM_channel_enhance == 2 else [3]
        channel = 'enhance' if ARGS.TSM_channel_enhance == 2 else 'shift'

        for kernelsize in kernelsize_list:
            conv = nn.Conv1d(in_channels=Mhw, out_channels=Mhw, kernel_size=kernelsize,
                             padding=math.floor(kernelsize / 2), groups=Mhw, bias=False)
            for n_div in [4, 8, 16]:
                fold = Mhw // n_div
                for block_position in ['start', 'middle', 'end', 'stochastic']:
                    for init_strategy in init_strategy_list:
                        conv = init_conv(conv, n_div, Mhw, block_position, fold, init_strategy, kernelsize)

                        plt.imshow(conv.weight.data.squeeze().permute(1, 0))
                        plt.savefig('_'.join([channel, 'kernelsize', str(kernelsize), 'n_div', str(n_div),
                                              'block_position', str(block_position), 'init_strategy',
                                              str(init_strategy)]) + '.jpg')

        sys.exit()


if __name__ == '__main__':
    pass
