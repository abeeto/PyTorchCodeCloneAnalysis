import torch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

print('CUDA availity: {}'.format(torch.cuda.is_available()))
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print('CUDA device count: {}'.format(device_count))
    for idx in range(device_count):
        print('Device {}: {}'.format(idx, torch.cuda.get_device_name(idx)))
