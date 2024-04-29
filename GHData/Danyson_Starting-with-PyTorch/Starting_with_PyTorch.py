# -*- coding: utf-8 -*-
"""


# Introduction

 This material in this Post draws from http://cs231n.github.io/python-numpy-tutorial/ and

https://github.com/kuleshov/cs228-material/blob/master/tutorials/python/cs228-python-tutorial.ipynb.

This material focuses mainly on PyTorch.

# PyTorch

[PyTorch](https://pytorch.org/) is an open source machine learning framework. At its core, PyTorch provides a few key features:

- A multidimensional **Tensor** object, similar to [numpy](https://numpy.org/) but with GPU accelleration.
- An optimized **autograd** engine for automatically computing derivatives
- A clean, modular API for building and deploying **deep learning models**



You can find more information about PyTorch by following one of the [official tutorials](https://pytorch.org/tutorials/) or by [reading the documentation](https://pytorch.org/docs/1.1.0/).

To use PyTorch, we first need to import the `torch` package.
"""

import torch
print(torch.__version__)

"""## Tensor Basics

### Creating and Accessing tensors

A `torch` **tensor** is a multidimensional grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. The number of dimensions is the **rank** of the tensor; the **shape** of a tensor is a tuple of integers giving the size of the array along each dimension.

We can initialize `torch` tensor from nested Python lists. We can access or mutate elements of a PyTorch tensor using square brackets.

Accessing an element from a PyTorch tensor returns a PyTorch scalar; we can convert this to a Python scalar using the `.item()` method:

*Note:The biggest difference between a numpy array and a PyTorch Tensor is that a PyTorch Tensor can run on either CPU or GPU.*
"""

# Create a rank 1 tensor from a Python list
a = torch.tensor([1, 2, 3]) # an one dimensional tensor 
print('Here is a:')
print(a)
print('type(a): ', type(a))     # exposes the type of our variable "a" which is a class of pytorch
print('rank of a: ', a.dim())   # dim shows the dimension of our tensor
print('a.shape: ', a.shape)     # shape says size of our tensor
# Access elements using square brackets
print()
print('a[0]: ', a[0])                           # access first element of tensor
print('type(a[0]): ', type(a[0]))               # shows the type of the first element in tensor as of pytorch class
print('type(a[0].item()): ', type(a[0].item())) # converts the type of the first element in tensor to python class

# Mutate elements using square brackets
a[1] = 10                              # mutation refers to the change in a data in the tensor as intentionaly or by accident
print()
print('a after mutating:')
print(a)                              # shows the mutated value in our tensor  at index 1

"""The example above shows a one-dimensional tensor; we can similarly create tensors with two or more dimensions:"""

# Create a two-dimensional tensor
b = torch.tensor([[1, 2, 3], [4, 5, 5]])          #This is a two-dimensional tensor array
print('Here is b:')
print(b)
print('rank of b:', b.dim())                      #Shows the dimension of tensor 'b'
print('b.shape: ', b.shape)                       #Shows the size of the 2D tensor 'b'

# Access elements from a multidimensional tensor
print()
print('b[0, 1]:', b[0, 1])                        #Prints 0th row 1st column element of tensor 'b'
print('b[1, 2]:', b[1, 2])                        #Prints 1st row 2nd column element of tensor 'b'

# Mutate elements of a multidimensional tensor
b[1, 1] = 100                                     #Mutate 1st row 1st column element of tensor 'b' with integer value '100'
print()
print('b after mutating:')
print(b)

"""1. Construct a tensor `c` of shape `(3, 2)` filled with zeros by initializing from nested Python lists.
2. Then set element `(0, 1)` to 10, and element `(1, 0)` to 100:
"""

c = None
################################################################################
#  Construction of a tensor c filled with all zeros, initializing from nested  #
# Python lists.                                                                #
################################################################################
c = torch.tensor([[0,0],[0,0],[0,0]])
print(c)
print()
print('c is a tensor: ', torch.is_tensor(c))     #checking is 'c' is a tensor
print('Correct shape: ', c.shape == (3, 2))      #Shows the size of tensor 'c'
print('All zeros: ', (c == 0).all().item() == 1) #Boolean checking of the elements in tensor 'c' are of integer '0's 
print()
################################################################################
# Mutating the element (0, 1) of c to 10, and element (1, 0) of c to 100.      #
################################################################################
c[0,1] = 10
c[1,0] = 100
print(c)
print('\nAfter mutating:')
print('Correct shape: ', c.shape == (3, 2))                    #Boolean checking the size of the tensor 'c' whether it has 3rows and 2columns
print('c[0, 1] correct: ', c[0, 1] == 10)                      #Boolean checking of whether 1st row 1st column element in tensor 'c' is of integer value 10
print('c[1, 0] correct: ', c[1, 0] == 100)                     #Boolean checking of whether 1st row 0th column element in tensor 'c' is of integer value 100
print('Rest of c is still zero: ', (c == 0).sum().item() == 4) #Boolean checking of whether the remaining 4 elements have value as 0 
# c == 0 returns True in places of the tensor whose values are 0 
# while sum() calculates the number of positions where tensor has value 0  
# and item() == 4 checks those places having value 0 in tensor are 4

"""### Tensor constructors

PyTorch provides many convenience methods for constructing tensors; this avoids the need to use Python lists. For example:

- [`torch.zeros`](https://pytorch.org/docs/1.1.0/torch.html#torch.zeros): Creates a tensor of all zeros
- [`torch.ones`](https://pytorch.org/docs/1.1.0/torch.html#torch.ones): Creates a tensor of all ones
- [`torch.rand`](https://pytorch.org/docs/1.1.0/torch.html#torch.rand): Creates a tensor with uniform random numbers

You can find a full list of tensor creation operations [in the documentation](https://pytorch.org/docs/1.1.0/torch.html#creation-ops).
"""

# Create a tensor of all zeros with 2 rows 3 columns 
a = torch.zeros(2, 3)
print('tensor of zeros:')
print(a)

# Create a tensor of all ones with 1 row and 2 columns
b = torch.ones(1, 2)
print('\ntensor of ones:')
print(b)

# Create a 3x3 identity matrix which is a 3X3 square matrix in which all 
# the elements of the principal diagonal are ones and all other elements are zeros.
c = torch.eye(3)
print('\nidentity matrix:')
print(c)

# Tensor of 4 rows and 5 columns having random values
d = torch.rand(4, 5)
print('\nrandom tensor:')
print(d)

e = None
################################################################################
# Creating a tensor of shape (2, 3, 4) filled entirely with 7, stored in e     #
################################################################################
e = torch.full((2, 3, 4), 7)                                   # using full function to fill value 7 in tensor 'e'
print(e)
print('e is a tensor:', torch.is_tensor(e))                    #Boolean checking of whether 'e' is a tensor
print('e has correct shape: ', e.shape == (2, 3, 4))           #Boolean checking of the size of tensor is equal to (2, 3, 4)
print('e is filled with sevens: ', (e == 7).all().item() == 1) #Boolean checking of whether the tensor 'e' elements are of value 7
# e == 7 returns all the places in tensor which have 7 as True 
# while  all() checks whether all the places having 7 in the tensor are true
# finally item() == 1 checks all items holding  True for value 7 in the tensor are the same

"""### Datatypes

In the examples above, you may have noticed that some of our tensors contained floating-point values, while others contained integer values.

PyTorch provides a [large set of numeric datatypes](https://pytorch.org/docs/1.1.0/tensor_attributes.html#torch-dtype) that you can use to construct tensors. PyTorch tries to guess a datatype when you create a tensor; functions that construct tensors typically have a `dtype` argument that you can use to explicitly specify a datatype.

Each tensor has a `dtype` attribute that you can use to check its data type:
"""

# Let torch choose the datatype
x0 = torch.tensor([1, 2])   # List of integers
x1 = torch.tensor([1., 2.]) # List of floats
x2 = torch.tensor([1., 2])  # Mixed list
print('dtype when torch chooses for us:')
print('List of integers:', x0.dtype)
print('List of floats:', x1.dtype)
print('Mixed list:', x2.dtype)

# Force a particular datatype
y0 = torch.tensor([1, 2], dtype=torch.float32)  # 32-bit float
y1 = torch.tensor([1, 2], dtype=torch.int32)    # 32-bit (signed) integer
y2 = torch.tensor([1, 2], dtype=torch.int64)    # 64-bit (signed) integer
print('\ndtype when we force a datatype:')
print('32-bit float: ', y0.dtype)
print('32-bit integer: ', y1.dtype)
print('64-bit integer: ', y2.dtype)

# Other creation ops also take a dtype argument
# torch.ones() creates tensor filled with 1 of different size and dtype
z0 = torch.ones(1, 2)  # Let torch choose for us
z1 = torch.ones(1, 2, dtype=torch.int16) # 16-bit (signed) integer
z2 = torch.ones(1, 2, dtype=torch.bool) # 8-bit (unsigned) integer
print('\ntorch.ones with different dtypes')
print('default dtype:', z0.dtype)
print('16-bit integer:', z1.dtype)
print('8-bit unsigned integer:', z2.dtype)

"""We can **cast** a tensor to another datatype using the [`.to()`](https://pytorch.org/docs/1.1.0/tensors.html#torch.Tensor.to) method; there are also convenience methods like [`.float()`](https://pytorch.org/docs/1.1.0/tensors.html#torch.Tensor.float) and [`.long()`](https://pytorch.org/docs/1.1.0/tensors.html#torch.Tensor.long) that cast to particular datatypes:"""

x0 = torch.eye(3, dtype=torch.int64)
x1 = x0.float()  # Cast to 32-bit float
x2 = x0.double() # Cast to 64-bit float
x3 = x0.to(torch.float32) # Alternate way to cast to 32-bit float
x4 = x0.to(torch.float64) # Alternate way to cast to 64-bit float
print('x0:', x0.dtype)
print('x1:', x1.dtype)
print('x2:', x2.dtype)
print('x3:', x3.dtype)
print('x4:', x4.dtype)

"""PyTorch provides several ways to create a tensor with the same datatype as another tensor:

- PyTorch provides tensor constructors such as [`torch.new_zeros()`](https://pytorch.org/docs/1.1.0/torch.html#torch.zeros_like) that create new tensors with the same shape and type as a given tensor
- Tensor objects have instance methods such as [`.new_zeros()`](https://pytorch.org/docs/1.1.0/tensors.html#torch.Tensor.new_zeros) that create tensors the same type but possibly different shapes
- The tensor instance method [`.to()`](https://pytorch.org/docs/1.1.0/tensors.html#torch.Tensor.to) can take a tensor as an argument, in which case it casts to the datatype of the argument.
"""

x0 = torch.eye(3, dtype=torch.float64)  # Shape (3, 3), dtype torch.float64
x1 = torch.zeros_like(x0)               # Shape (3, 3), dtype torch.float64
x2 = x0.new_zeros(4, 5)                 # Shape (4, 5), dtype torch.float64
x3 = torch.ones(6, 7).to(x0)            # Shape (6, 7), dtype torch.float64)
print('x0 shape is %r, dtype is %r' % (x0.shape, x0.dtype))
print('x1 shape is %r, dtype is %r' % (x1.shape, x1.dtype))
print('x2 shape is %r, dtype is %r' % (x2.shape, x2.dtype))
print('x3 shape is %r, dtype is %r' % (x3.shape, x3.dtype))

"""Create a 64-bit floating-point tensor of shape (6,) (six-element vector) filled with evenly-spaced values between 10 and 20."""

x = None
##############################################################################
# x contains a six-element vector of 64-bit floating-bit values,              #
# evenly spaced between 10 and 20.                                           #
##############################################################################
x = torch.linspace(10, 20, steps=6, dtype=torch.float64 ) 
# the value set in steps divides the the number between start=10 and end=20 to
# equal discrete intervals. 
print('x is a tensor: ', torch.is_tensor(x))
print('x has correct shape: ', x.shape == (6,))
print('x has correct dtype: ', x.dtype == torch.float64)
y = [10, 12, 14, 16, 18, 20]
correct_vals = all(a.item() == b for a, b in zip(x, y))# checks and return true 
# if all the values of x is same as y
print('x has correct values: ', correct_vals)

"""Even though PyTorch provides a large number of numeric datatypes, the most commonly used datatypes are:

- `torch.float32`: Standard floating-point type; used to store learnable parameters, network activations, etc. Nearly all arithmetic is done using this type.
- `torch.int64`: Typically used to store indices
- `torch.bool`: Typically used to store boolean values, where 0 is false and 1 is true.
- `torch.float16`: Used for mixed-precision arithmetic, usually on NVIDIA GPUs with [tensor cores](https://www.nvidia.com/en-us/data-center/tensorcore/). 

Note that PyTorch version 1.2.0 introduced a new `torch.bool` datatype for holding boolean values. However for earlier versions (including 1.1.0)  `torch.uint8` used to hold boolean values instead. We used PyTorch version 1.5.1+cu101 for this post.

## Tensor indexing

We have already seen how to get and set individual elements of PyTorch tensors. PyTorch also provides many other ways of indexing into tensors. Getting comfortable with these different options makes it easy to modify different parts of tensors with ease.

### Slice indexing

Similar to Python lists and numpy arrays, PyTorch tensors can be **sliced** using the syntax `start:stop` or `start:stop:step`. The `stop` index is always non-inclusive: it is the first element not to be included in the slice.

Start and stop indices can be negative, in which case they count backward from the end of the tensor.
"""

a = torch.tensor([0, 11, 22, 33, 44, 55, 66])
print(0, a)        # (0) Original tensor
print(1, a[2:5])   # (1) Elements between index 2 and 5
print(2, a[2:])    # (2) Elements after index 2
print(3, a[:5])    # (3) Elements before index 5
print(4, a[:])     # (4) All elements
print(5, a[1:5:2]) # (5) Every second element between indices 1 and 5
print(6, a[:-1])   # (6) All but the last element
print(7, a[-4::2]) # (7) Every second element, starting from the fourth-last

"""For multidimensional tensors, you can provide a slice or integer for each dimension of the tensor in order to extract different types of subtensors:"""

# Create the following rank 2 tensor with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = torch.tensor([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print('Original tensor:')
print(a)
print('shape: ', a.shape)

# Get row 1, and all columns. 
print('\nSingle row:')
print(a[1, :])
print(a[1])  # Gives the same result; we can omit : for trailing dimensions
print('shape: ', a[1].shape)

print('\nSingle column:')
print(a[:, 1])
print('shape: ', a[:, 1].shape)

# Get the first two rows and the last three columns
print('\nFirst two rows, last three columns:')
print(a[:2, -3:])
print('shape: ', a[:2, -3:].shape)

# Get every other row, and columns at index 1 and 2
print('\nEvery other row, middle columns:')
print(a[::2, 1:3]) 
print('shape: ', a[::2, 1:3].shape)

"""There are two common ways to access a single row or column of a tensor: using an integer will reduce the rank by one, and using a length-one slice will keep the same rank. Note that this is different behavior from MATLAB."""

# Create the following rank 2 tensor with shape (3, 4)
a = torch.tensor([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print('Original tensor')
print(a)

row_r1 = a[1, :]    # Rank 1 view of the second row of a  
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print('\nTwo ways of accessing a single row:')
print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)

# We can make the same distinction when accessing columns::
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print('\nTwo ways of accessing a single column:')
print(col_r1, col_r1.shape)
print(col_r2, col_r2.shape)

"""Slicing a tensor returns a **view** into the same data, so modifying it will also modify the original tensor. To avoid this, you can use the `clone()` method to make a copy of a tensor."""

# Create a tensor, a slice, and a clone of a slice
a = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
b = a[0, 1:]
c = a[0, 1:].clone()
print('Before mutating:')
print(a)
print(b)
print(c)

a[0, 1] = 20  # a[0, 1] and b[0] point to the same element
b[1] = 30     # b[1] and a[0, 2] point to the same element
c[2] = 40     # c is a clone, so it has its own data
print('\nAfter mutating:')
print(a)
print(b)
print(c)

print(a.storage().data_ptr() == c.storage().data_ptr())

"""Your turn: practice indexing tensors with slices"""

# We will use this helper function to check your results
def check(orig, actual, expected):
  expected = torch.tensor(expected)
  same_elements = (actual == expected).all().item() == 1
  same_storage = (orig.storage().data_ptr() == actual.storage().data_ptr())
  return same_elements and same_storage

# Create the following rank 2 tensor of shape (3, 5)
# [[ 1  2  3  4  5]
#  [ 6  7  8  9 10]
#  [11 12 13 14 15]]
a = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 8, 10], [11, 12, 13, 14, 15]])

b, c, d, e = None, None, None, None
##############################################################################
# Extract the last row of a, and store it in b; it should have rank 1.       #
##############################################################################
b = a[-1:]

print('b correct:', check(a, b, [11, 12, 13, 14, 15]))

##############################################################################
# Extract the third col of a, and store it in c; it should have rank 2       #
##############################################################################
c = a[:,2:3]

print('c correct:', check(a, c, [[3], [8], [13]]))

##############################################################################
# Use slicing to extract the first two rows and first three columns          #
# from a; store the result into d.                                           #
##############################################################################
d = a[:2,0:3]

print('d correct:', check(a, d, [[1, 2, 3], [6, 7, 8]]))

##############################################################################
# Use slicing to extract a subtensor of a consisting of rows 0 and 2         #
# and columns 1 and 4. Store result into e                                   #
##############################################################################
e = a[0::2,1::3] # in row side start from 0 and take 2 steps and for column
# start from 1 and take 3 steps,the '::' enables you to take steps.
print('e correct:', check(a, e, [[2, 5], [12, 15]]))

"""So far we have used slicing to **access** subtensors; we can also use slicing to **modify** subtensors by writing assignment expressions where the left-hand side is a slice expression, and the right-hand side is a constant or a tensor of the correct shape:"""

a = torch.zeros(2, 4, dtype=torch.int64)
a[:, :2] = 1 # choose 0 to 1 columns and fill with value 1
a[:, 2:] = torch.tensor([[2, 3], [4, 5]]) # choose columns after column 1 and
# fill with the input tensor of shape 2,2 into the tensor 'a' of shape 2,4
print(a)

