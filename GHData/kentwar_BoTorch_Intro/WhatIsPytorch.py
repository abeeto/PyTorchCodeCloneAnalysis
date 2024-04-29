## What is pytorch?
## by Paul Kent
##

## 1. A replacement for NumPy to use the power of GPUs
## 2. a deep learning research platform that provides maximum flexibility and speed
#%%

import torch

#################### basic functionality #######################################

## Create a tensor with dimensions 5x3

x1 = torch.empty( 5 , 3 )
print( 'empty' , x1 )

#x[0,1] = 5

### Bear in mind that pytorch.empty() assigns a piece of memory 
### and grabs whatever is in that memory block. It is not the same as
### torch.zeros()
#%%

x2 = torch.zeros( 5, 3 )
print( 'zeros' , x2 )

x3 = torch.rand( 5, 3 )
print( 'random' , x3 )

#%%

### Create a tensor from data
### torch.tensor( data )

my_data = [ [ 5 , 6 ] , [ 7 , 5 ] ]
x4 = torch.tensor( my_data )
print( 'my_data' , x4 )
# %%

### You can get a tensors size in a familiar way:

print( x4.size() )
# %%

#Tensor addition requires two tensors of the same size

print( x1 )
print( x3 )
print( x1 + x3 )
#print( x1 + x4 ) # Will induce error

# %%

## You can also perform addition in place
print( x1 )
x1.add_( x1 ) 
# Any operation that mutates a tensor in-place is post-fixed with an _. 
# For example: x.copy_(y), x.t_(), will change x.
print( x1 ) 
# %%

## Accessing cells is exactly like numpy arrays
print( x3 )
first_col = x3[ : , 0 ]
print( first_col )
# %%

## Resizing is familiar too.

x5 = torch.randn( 4 , 4 )
print( x5 )
print( x5.size() )
#%%
x5_1 = x5.view( 8 , -1 )
print( x5_1 )
print( x5_1.size() )
#%%

print( x5.view( -1 , 16 ))
# %%

## converting between numpy arrays and tensors is also easy
import numpy as np

npx = np.ones( 10 )
x6 = torch.from_numpy( npx )
print( npx )
print( x6 )
# %%

a = torch.ones( 5 )
b = a.numpy()
print( a )
print( b )

# %%
a.add_( 1 )
print( a )
print( b )
# %%
