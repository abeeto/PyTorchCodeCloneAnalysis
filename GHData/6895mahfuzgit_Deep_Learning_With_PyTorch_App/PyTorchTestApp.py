# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 17:09:09 2021

@author: Mahfuz_Shazol
"""


import torch
import numpy as np

x = torch.tensor(4.0)
# print(x)
# print(x.dtype)

vectorTensor = torch.tensor([1,2,3,4.,5,6])
# print(vectorTensor)
# print(vectorTensor.dtype)


matrixTensor = torch.tensor([
                           [
                            [1,2,3,4],
                            [5,6,7,8]
                            ],
                           [
                            [9,7,5,3],
                            [1,2,3,7.]
                            ]
    ])
# print(matrixTensor)
# print(matrixTensor.dtype)
# print(matrixTensor.shape)



x=torch.tensor(4.)
w=torch.tensor(5.,requires_grad=True)
b=torch.tensor(6.,requires_grad=True)

y=w*x+b
#print(y)
y.backward()

# print('backword value of x:',x.grad)
# print('backword value of w:',w.grad)
# print('backword value of b:',b.grad)

#Create a tensor with value 

tFull=torch.full((3,2),0)
t1=torch.tensor([[ 0,1],
        [ 0,2],
        [ 0,3],
        [ 0,4],
        [ 0,5],
        [ 0,6],
        [ 0,7],
        [ 0,8],
        [ 0,9],
        ])

#concate
tConcat=torch.cat((tFull,t1))


tSine=torch.sin(t1)
#print(tSine)

tReshape=t1.reshape(3,3,2)
#print(tReshape)






x=np.array([[1,2],[3,4.] ])
print('Numpy Array Type',x.dtype)

y=torch.from_numpy(x)
print('Torch Array Type',y.dtype)

z=y.numpy()
print('Torch Array To Numpy Type',z.dtype)












