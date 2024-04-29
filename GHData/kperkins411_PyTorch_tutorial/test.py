import numpy as np

import torch
import numpy as np

torch.manual_seed(4)
x = torch.rand(3,4)
y = torch.rand(3,4)
print('y', y)
_, argmax = x.max(-1)
print('argmax', argmax)
y[np.arange(3), argmax] = 3
print('y', y)



ar = np.arange(28).reshape(4,7)
# data1 = ar(:,names = 'bob')

name = "layer 223"
l = "layer"
print( int(name[len(l):]))
print( name[:len(l)])

print (int(3//3))


str1 ="layer1.0.conv1.weight"
tok = str1.split('.')[0]

str2="layer1"
tok2 = str2.split('.')[0]
pass


