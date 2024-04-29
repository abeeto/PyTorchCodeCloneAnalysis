import torch
from torch2mindspore import *
from module_test import ModuleTest, MODULE_TESTS
import time
import argparse
import re
import runpy
import traceback
from termcolor import colored
from ms_converters import *

from models.mwcnn_trt import MWCNN_trt as MWCNN
# from models.sr_hr_mwcnn import *
from models.sr_hrnet import *
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--n_resblocks', type=int, default=16)
parser.add_argument('--n_feats', type=int, default=64)
parser.add_argument('--n_colors', type=int, default=1)
parser.add_argument('--res_scale', type=int, default=1)
args = parser.parse_args()

class Custom(torch.nn.Module):
    def __init__(self):
        super(Custom, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, 1)
        self.conv1 = torch.nn.Conv2d(3, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        _ = self.conv(x)
        _ = self.conv1(_)
        _ = x + _
        _ = self.relu(_)
        t = self.conv2(_)
        t = self.relu(t)
        r = torch.cat([_, t])
        return r * 100


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], enabled=True)
@add_module_test(torch.float32, torch.device('cuda'), [(1, 1, 224, 224)], enabled=True)
def test_custom():
    # return MWCNN(args)
    # return get_sr_model()
    return get_sr_hrnet_model()


def run(self):
    # create module
    module = self.module_fn()
    module = module.to(self.device)
    module = module.type(self.dtype)
    module = module.eval()

    # create inputs for conversion
    inputs_conversion = ()
    for shape in self.input_shapes:
        inputs_conversion += (torch.zeros(shape).to(self.device).type(self.dtype), )


    # convert module
    module_ms = torch2mindspore(module, inputs_conversion, max_workspace_size=1 << 20,  **self.torch2ms_kwargs)

    # create inputs for torch/ms.. copy of inputs to handle inplace ops
    inputs = ()
    for shape in self.input_shapes:
        inputs += (torch.randn(shape).to(self.device).type(self.dtype), )
    inputs_ms = tuple([tensor.clone() for tensor in inputs])


    # test output against original
    outputs = module(*inputs)
    outputs_ms = module_ms(*inputs_ms)

    if not isinstance(outputs, tuple):
        outputs = (outputs, )

    if not isinstance(outputs_ms, tuple):
        outputs_ms = (outputs_ms, )

    # compute max error
    max_error = 0
    for i in range(len(outputs)):
        max_error_i = 0
        if outputs[i].dtype == torch.bool:
            max_error_i = torch.sum(outputs[i] ^ outputs_ms[i])
        else:
            max_error_i = torch.max(torch.abs(outputs[i] - outputs_ms[i]))

        if max_error_i > max_error:
            max_error = max_error_i


    if max_error > 1:
        print(outputs)
        print(outputs_ms)
    # benchmark pytorch throughput
    torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(50):
        outputs = module(*inputs)
    torch.cuda.current_stream().synchronize()
    t1 = time.time()

    fps = 50.0 / (t1 - t0)

    # benchmark tensorrt throughput
    torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(50):
        outputs = module_ms(*inputs)
    torch.cuda.current_stream().synchronize()
    t1 = time.time()

    fps_ms = 50.0 / (t1 - t0)

    # benchmark pytorch latency
    torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(50):
        outputs = module(*inputs)
        torch.cuda.current_stream().synchronize()
    t1 = time.time()

    ms = 1000.0 * (t1 - t0) / 50.0

    # benchmark tensorrt latency
    torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(50):
        outputs = module_ms(*inputs)
        torch.cuda.current_stream().synchronize()
    t1 = time.time()

    ms_ms = 1000.0 * (t1 - t0) / 50.0

    # jay test
    print('-------big size--------')
    x = [torch.rand((1, 1, 512, 512)).cuda().type(torch.float32)]
    ms_y = module_ms(*x)
    print('-------end--------', ms_y.shape)

    return max_error, fps, fps_ms, ms, ms_ms


if __name__ == '__main__':
    MODULE_TESTS = [MODULE_TESTS[-1]]  # test last one
    for test in MODULE_TESTS:
        # filter by module name
        name = test.module_name()
        print(f' test name: {name}')
        # run(test)
        max_error, fps, fps_ms, ms, ms_ms = run(test)

        # write entry
        line = '| %s | %s | %s | %s | %.2E | %.3g | %.3g | %.3g | %.3g |' % (name, test.dtype.__repr__().split('.')[-1], str(test.input_shapes), str(test.torch2ms_kwargs), max_error, fps, fps_ms, ms, ms_ms)

        print(line)


