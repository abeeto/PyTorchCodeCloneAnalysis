# script to detect device type when using PyTorch

import torch

if(torch.cuda.is_available()):
    print("This GPU is available for CUDA and PyTorch")
    print("##########################################")
    device = torch.cuda.current_device()
    print("Current Device: " , device)
    print("Device Memory Location: " , torch.cuda.device(device))
    print("Count Device: " , torch.cuda.device_count())
    print("Device Name: " , torch.cuda.get_device_name(device))
else:
    torch.device("cpu")
    print("This GPU is not available for CUDA and PyTorch")
    print("##########################################")