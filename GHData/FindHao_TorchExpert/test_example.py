
import torch
import torch.nn.functional as F
import argparse
from typing import Dict, List
from torch import profiler

from torchexpert import TorchExpert

torchexpert = TorchExpert()

def run_conv2d(input_shape, weight_shape, other_args, profile_folder):
    input = torch.ones(input_shape, dtype=torch.float32, device='cuda')
    weight = torch.ones(weight_shape, dtype=torch.float32, device='cuda')
    bias = other_args[0]
    stride = other_args[1]
    padding = other_args[2]
    dilation = other_args[3]
    groups = other_args[4]
    # warmup
    for i in range(11):
        x = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    for i in range(1000):
        x = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    return x


def run():
    input = torch.ones(input_shape, dtype=torch.float32, device='cuda')
    weight = torch.ones(weight_shape, dtype=torch.float32, device='cuda')
    bias = other_args[0]
    stride = other_args[1]
    padding = other_args[2]
    dilation = other_args[3]
    groups = other_args[4]
    output = F.conv2d(input, weight, bias, stride, padding, dilation, groups).to('cpu')
    

def profile(input_shape, weight_shape, other_args):
    torchexpert.profile(run)
    # torchexpert.analyze_json = True
    torchexpert.model_name = "conv2d"
    torchexpert.output_csv_file = "conv2d.csv"
    torchexpert.analyze("./logs/")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    SUPPORT_BATCHSIZE_LIST = ['32', '64']
    parser.add_argument("--bs", choices=SUPPORT_BATCHSIZE_LIST,
                        help="Specify batch size to the test.")
    parser.add_argument("--profile-folder", default="./logs",
                        help="Save profiling model traces to this directory.")
    args, extra_args = parser.parse_known_args()
    torch.backends.cudnn.benchmark = False
    if args.bs == '64':
        input_shape = (64, 224, 112, 112)
        other_args = [None, (2, 2), (1, 1), (1, 1), 2]
    else:
        input_shape = (32, 224, 56, 56)
        other_args = [None, (1, 1), (1, 1), (1, 1), 2]
    weight_shape = (224, 112, 3, 3)
    profile(input_shape, weight_shape, other_args)
