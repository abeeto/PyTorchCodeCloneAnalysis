import torch
import torch.nn as nn


def tensor(v):
    if isinstance(v, torch.Tensor):
        return v.cuda() if gpu_available() else v
    else:
        return torch.tensor(v).cuda() if gpu_available() else torch.tensor(v)


def zeros(*kwargs):
    v = torch.zeros(kwargs)
    return v.cuda() if gpu_available() else v


def sequence_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1).lt(lengths.unsqueeze(1))


def pad_zeros(value, full_size, dim=0):
    if full_size == value.shape[0]:
        return value
    padding = [0] * (value.dim() * 2)
    padding[-dim*2-1] = full_size - value.shape[dim]
    padded_value = nn.functional.pad(value, padding)
    return padded_value


def softmax_mask(val, mask):
    return -1E18 * (1 - mask.float()) + val
    

def gpu_available():
    return torch.cuda.is_available()


def use_last_gpu():
    device = torch.cuda.device_count() - 1
    torch.cuda.set_device(device)