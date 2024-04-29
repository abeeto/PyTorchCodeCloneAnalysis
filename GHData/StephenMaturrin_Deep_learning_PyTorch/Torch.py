from __future__ import print_function
import torch
import gzip # pour décompresser les données

import numpy # pour pouvoir utiliser des matrices
import matplotlib.pyplot as plt # pour l'affichage
import torch,torch.utils.data
import pickle  # pour désérialiser les données


x = torch.Tensor(5, 3)
y = torch.rand(5, 3)

print(x)
print(y)
print(x.add(y))

result = torch.Tensor(5,3)



torch.add(x,y,out=result)

print( result)