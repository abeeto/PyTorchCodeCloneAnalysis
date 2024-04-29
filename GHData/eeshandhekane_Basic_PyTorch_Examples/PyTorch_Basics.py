# Dependencies
import numpy as np
import torch


"""
# Create range of numbers and reshape. Convert to numpy tensor and then back to array
arr = np.arange(8)
print arr
print arr.shape
mat = arr.reshape((2, 4))
print mat
print mat.shape
torch_mat = torch.from_numpy(mat)
print torch_mat
print torch_mat.shape
numpy_mat = torch_mat.numpy()
print numpy_mat 
print numpy_mat.shape
# Results
[0 1 2 3 4 5 6 7]
(8,)
[[0 1 2 3]
 [4 5 6 7]]
(2, 4)

 0  1  2  3
 4  5  6  7
[torch.LongTensor of size 2x4]

(2L, 4L)
[[0 1 2 3]
 [4 5 6 7]]
(2, 4)
"""


"""
# The previous example created tensor and arrays of int (long) type. We now do the same for float sensor
data = [-1, -3, 3, 1]
tens = torch.FloatTensor(data)
print('1')
print tens
print('2')
print np.abs(data)
print torch.abs(tens)
print('3')
print np.mean(data)
print torch.mean(tens)
print('4')
print np.tanh(data) # sin, cos, tan, sinh, cosh, tanh
print torch.tanh(tens)
# Results
1

-1
-3
 3
 1
[torch.FloatTensor of size 4]

2
[1 3 3 1]

 1
 3
 3
 1
[torch.FloatTensor of size 4]

3
0.0
0.0
4
[-0.76159416 -0.99505475  0.99505475  0.76159416]

-0.7616
-0.9951
 0.9951
 0.7616
[torch.FloatTensor of size 4]
"""


"""
# Matrix Multiplication
mat1 = np.array([[1, 2], [3, 4], [5, 6]])
mat2 = np.array([[1, 2, 3], [4, 5, 6]])
tens1 = torch.FloatTensor(mat1)
tens2 = torch.FloatTensor(mat2)
print('1')
print tens1
print tens1.shape
print tens2
print tens2.shape
print('2')
numpy_prod = np.matmul(mat1, mat2)
torch_prod = torch.mm(tens1, tens2)
print numpy_prod
print numpy_prod.shape
print torch_prod
print torch_prod.shape
print('3')
print mat1.dot(mat2)
#print tens1.dot(tens2) # DOES NOT WORK!!
# Results
1

 1  2
 3  4
 5  6
[torch.FloatTensor of size 3x2]

(3L, 2L)

 1  2  3
 4  5  6
[torch.FloatTensor of size 2x3]

(2L, 3L)
2
[[ 9 12 15]
 [19 26 33]
 [29 40 51]]
(3, 3)

  9  12  15
 19  26  33
 29  40  51
[torch.FloatTensor of size 3x3]

(3L, 3L)
3
[[ 9 12 15]
 [19 26 33]
 [29 40 51]]
"""


# Dependencies
from torch.autograd import Variable


"""
# Define variables with values, setting requires_grad to true. Variables in PyTorch are dynamic as opposed to TensorFlow's static variables
tens = torch.FloatTensor(np.array([[1, 2, 3], [4, 5, 6]]))
print('########## 1 ###########')
print tens
print tens.shape
# Initialize variable with tensor value, and enable for gradients
var = Variable(tens, requires_grad = True) 
print('########## 2 ###########')
print var
#print var.shape # NOT DEFINED!
# Compute a result
tens_res = torch.mean(tens*tens) # Element-wise product
var_res = torch.mean(var*var)
print('########## 3 ###########')
print tens_res
print var_res
# This answer is (1/6)*(sum of e_{ij}^2). Gradient of this will be (1/6)*2*var
print('########## 4 ###########')
print var_res.grad
print var.grad
var_res.backward() # This only passes back the gradients, does not change weights
print('########## 5 ###########')
print var_res.grad
print var.grad # It is indeed 1/3 of var
print('########## 6 ###########')
print var_res.data
print var.data
print var.grad.data.numpy() # Example to show how to retrieve data from the arrays
# Results
########## 1 ###########

 1  2  3
 4  5  6
[torch.FloatTensor of size 2x3]

(2L, 3L)
########## 2 ###########
Variable containing:
 1  2  3
 4  5  6
[torch.FloatTensor of size 2x3]

########## 3 ###########
15.1666666667
Variable containing:
 15.1667
[torch.FloatTensor of size 1]

########## 4 ###########
None
None
########## 5 ###########
None
Variable containing:
 0.3333  0.6667  1.0000
 1.3333  1.6667  2.0000
[torch.FloatTensor of size 2x3]

########## 6 ###########

 15.1667
[torch.FloatTensor of size 1]


 1  2  3
 4  5  6
[torch.FloatTensor of size 2x3]

[[ 0.33333334  0.66666669  1.        ]
 [ 1.33333337  1.66666675  2.        ]]
"""


# Dependencies
import torch.nn.functional as F


"""
# Generate data via linspace
x = torch.linspace(-5, 5, 10) 
print('########## 1 ###########')
print x # It is a float tensor of size 10
x = Variable(x)
# Find the values of all activations and plot them
print('########## 2 ###########')
y_relu_var = F.relu(x)
print y_relu_var
y_sigmoid_var = F.sigmoid(x)
print y_sigmoid_var
y_tanh_var = F.tanh(x)
print y_tanh_var
y_softplus_var = F.softplus(x)
print y_softplus_var
y_softmax_var = F.softmax(x)
print y_softmax_var
# Results
########## 1 ###########

-5.0000
-3.8889
-2.7778
-1.6667
-0.5556
 0.5556
 1.6667
 2.7778
 3.8889
 5.0000
[torch.FloatTensor of size 10]

########## 2 ###########
Variable containing:
 0.0000
 0.0000
 0.0000
 0.0000
 0.0000
 0.5556
 1.6667
 2.7778
 3.8889
 5.0000
[torch.FloatTensor of size 10]

Variable containing:
 0.0067
 0.0201
 0.0585
 0.1589
 0.3646
 0.6354
 0.8411
 0.9415
 0.9799
 0.9933
[torch.FloatTensor of size 10]

Variable containing:
-0.9999
-0.9992
-0.9923
-0.9311
-0.5047
 0.5047
 0.9311
 0.9923
 0.9992
 0.9999
[torch.FloatTensor of size 10]

Variable containing:
 0.0067
 0.0203
 0.0603
 0.1730
 0.4535
 1.0090
 1.8397
 2.8381
 3.9092
 5.0067
[torch.FloatTensor of size 10]

Variable containing:
 0.0000
 0.0001
 0.0003
 0.0009
 0.0026
 0.0079
 0.0239
 0.0727
 0.2208
 0.6708
[torch.FloatTensor of size 10]
"""