import os
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()
    return args

def set_device(local_rank=None):
    """The operation for set torch device.
    Args:
        local_rank (int): The local rank. Defaults to None.
    Returns:
        torch.device: The torch device.
    """
    device = None

    if torch.cuda.is_available():
        if local_rank is not None:
            device = torch.device('cuda:{}'.format(local_rank))
        else:
            device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    return device

def set_model(model, device, distributed=False):
    """The operation for set model's distribution mode.
    Args:
        model (nn.Module): The model.
        device (torch.device): The torch device.
        distributed (bool, optional): The option for distributed. Defaults to False.
    Raises:
        ValueError: If distributed gpu option is true, the gpu device should cuda.

    Returns:
        nn.Module: The model.
    """
    is_cuda = torch.cuda.is_available()

    if distributed:
        if is_cuda:
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model,device_ids=[device],output_device=[device])
        else:
            raise ValueError(
                'If in cpu or mps mode, distributed option should be False.')
    else:
        model = model.to(device)

        if is_cuda and torch.cuda.device_count()>1:
            model = nn.parallel.DataParallel(model)

    return model

def main():
    args = parse_args()

    distributed = False

    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        distributed = True
        print('This is {} rank of {} process'.format(
            os.environ['RANK'], os.environ['WORLD_SIZE']))
    
    device = set_device(args.local_rank if distributed else None)    

    if distributed:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=int(
                os.environ['WORLD_SIZE']), rank=int(os.environ['RANK']))
        
        if torch.distributed.is_initialized():
                print('Distribution is initalized.')
        else:
            print('Distirbution is not initalized.')

    model = torchvision.models.resnet18(pretrained=False)
    model = set_model(model,device,distributed=distributed)
    print('Get model.')

if __name__ == "__main__":
    main()
