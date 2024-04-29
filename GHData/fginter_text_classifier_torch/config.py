import torch
import torch.cuda
import sys

cuda=False #global variable indicating whether we run on cuda
torch_mod=torch #module to call the various tensor constructors on - will be set to torch.cuda if we run on GPU

# Global config
def set_cuda(use_cuda=True,silent=False):
    """
    Enables cuda if use_cuda is True and cuda is available. If silent is True, no warnings are issued
    """
    global cuda, torch_mod
    
    if use_cuda and not torch.cuda.is_available() and not silent:
        print("Usage of CUDA requested, but CUDA does not seem available. Try to check with 'nvidia_smi'.",file=sys.stderr)
    if use_cuda and torch.cuda.is_available():
        #we are good to go on GPU
        cuda=True
        torch_mod=torch.cuda
    else:
        cuda=False
        toch_mod=torch
