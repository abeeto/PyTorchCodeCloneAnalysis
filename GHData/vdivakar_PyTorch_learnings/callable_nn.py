import torch
import torch.nn as nn

'''
Instead of directly calling forward() method of nn.Module instance,
call the actual instance... i.e.
[x ] self.conv1.forward(tensor)
[ok] self.conv1(tensor)
'''

in_features = torch.tensor([1,2,3,4], dtype=torch.float32)
weight_matrix = torch.tensor([
    [1,2,3,4],
    [2,3,4,5],
    [3,4,5,6]
], dtype=torch.float32)

print("MatMul: ",weight_matrix.matmul(in_features))
print("Transpose: ", in_features.matmul(weight_matrix.t()))

'''Creates a weight matrix of shape (3,4)'''
fc = nn.Linear(in_features=4, out_features=3)
'''nn modules are Callable
__call__() method which inturn invokes forward() method'''
out = fc(in_features)
print("out: ",out)

'''The weight matrix above inside is intialized with random weight.
Hence different outputs'''
fc.weight = nn.Parameter(weight_matrix)
out2 = fc(in_features)
print("out2: ",out2)

'''setting bias off for exact values'''
fc = nn.Linear(in_features=4, out_features=3, bias=False)
fc.weight = nn.Parameter(weight_matrix)
out3 = fc(in_features)
print("out3: ",out3)