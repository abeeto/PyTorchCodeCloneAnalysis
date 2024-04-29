import torch

x = torch.empty(2, 2) #creating empty tensor

x = torch.rand(2, 2) # creating random numbers in 2x2 matrix

x = torch.zeros(2, 2) # creating 2x2 matrix with 0,like in numpy

x = torch.ones(2, 2, dtype = torch.float16) # can change dtype with dtype = torch.float16 or torch.int or torch.double etc.
print(x.dtype) # gives data type of x
print(x.size()) # gives size of x

x = torch.tensor([2.5, 0.1]) # creating a tensor

#%% Basic Operations
x = torch.rand(2, 2)
y = torch.rand(2, 2)

z = x + y
z = torch.add(x,y) # same thing -> z = x + y
y.add_(x) # modify the y and same thing -> x + y

z = x - y
z = torch.sub(x, y) # same thing -> z = x - y

z = x * y
z = torch.mul(x, y) # same thing -> z = x * y

z = x / y
z = torch.div(x, y) # same thing -> z = x / y
#%% Slicing Operations
x = torch.rand(5, 3)
print(x[:, 0]) # All rows and only the column zero will be printed
print(x[1, :]) # 1st row and all columns will be printed
print(x[1, 1]) # The element at [1, 1] position will be printed
print(x[1, 1].item()) # This method get the actual value BUT you can use this method for only one element
#%% Re-shape
x = torch.rand(4 ,4)
y = x.view(16) # You can use view() method to reshape, the size must be correct
y = x.view(-1, 8) # When we write the -1, the pytorch will automatically determine right size for it
#%% Converting Operations
import numpy as np

a = torch.ones(5)
b = a.numpy() # From tensor to numpy array converting
print(type(b))
a.add_(1) # It modified also b, because they both point to the same memory location - BE CAREFUL-

a = np.ones(5)
b = torch.from_numpy(a) # From numpy array to tensor converting

a += 1 # Again it modified also b, this happens only if your tensor is on the GPU

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device) # Creates a tensor and put it on GPU
    y = torch.ones(5) # Creates a tensor
    y = y.to(device) # Move it to GPU
    z = x + y # This will be performed on the GPU and might be much faster
    z.numpy() # It gives an error, because numpy can only handle on CPU tensor
    z = z.to("cpu") # You can move it back to CPU