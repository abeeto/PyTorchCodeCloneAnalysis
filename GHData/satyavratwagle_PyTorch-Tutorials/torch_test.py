import torch
import numpy as np

empty_matrix        = torch.empty(5,3)
random_matrix       = torch.rand(5,3)
long_zeros          = torch.zeros(5,3,dtype = torch.long)
torch_tensor        = torch.tensor([5.5,3])             # Tensor directly from array

to_be_sampled       = torch.ones(5,3)
sampled_random      = torch.randn_like(to_be_sampled)   # Matrix of the same shape
sampled_random_size = sampled_random.size()             # Returns a tuple

# Operations

x = torch.ones(3,3)
y = torch.ones(3,3)
to_be_an_item = torch.randn(1)

addition        = torch.add(x,y)
addition        = y.add(x)                                  # Addition which does not change y
addition        = y.add_(x)                                 # Addition which does changes y
reshape_tensor  = y.view(-1,9)                              # Reshape Tensor

pyvariable      = to_be_an_item.item()                      # Saves a single scalar variable as a Python variable
y_nparray       = y.numpy()                                 # Saves a tensor as a Numpy array --> Is not a copy! Holds actual value of Tensor!
y_torchtensor   = torch.from_numpy(y_nparray)               # Converts a Numpy array to a Torch Tensor --> All values are shared between instances

#print(y_torchtensor)

# Move to GPU

if torch.cuda.is_available():
    print("Using GTX 1050...")
    gpu = torch.device("cuda")
    
    # Create variables on GPU : Cannot be read as Numpy Arrays while on GPU
    create_on_gpu = torch.rand(5,3,device = gpu)
    send_to_gpu                 = y.to(gpu)
    send_to_gpu_another_method  = y.to("cuda")
    
    