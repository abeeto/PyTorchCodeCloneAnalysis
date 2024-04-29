import torch
from torch.nn import Linear
import numpy as np

inputs=torch.randn(3)
print("inputs:\n",inputs.shape)
linear=Linear(in_features=3,out_features=3)
print("weights:\n",linear.weight)
print("bias:\n",linear.bias)
print("linear with torch:\n",linear(inputs))
print("linear with manualy:\n",linear.weight*inputs+linear.bias)

linear2=Linear(in_features=10,out_features=5,bias=True)
print("Linear2 weights: ",linear2.weight)
linear3=Linear(in_features=5,out_features=1)
print("Linear3 weights: ",linear3.weight)