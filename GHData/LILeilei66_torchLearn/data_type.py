"""
torch.Tensor(), 与 torch.from_numpy()
-------------------------------------
**torch.Tensor():**
    本质为 torch.FloatTensor().
    与 numpy 共享内存, 但当 numpy 的数据类型与 Tensor 的数据类型不一致时, 数据会被复制，不会共享内存.
**torch.from_numpy():**
    从 numpy 继承数据类型.
    与原 numpy 数据共享内存.
**torch.tensor():**
    不论输入的类型是什么，t.tensor()都会进行数据拷贝，不会共享内存.
"""

import numpy as np
import torch
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
f = pd.read_csv('E:\TorchLearn\label_position.csv')
a = f.iloc[1,1:].values.astype(np.float64)

b = torch.Tensor(a)
c = torch.from_numpy(a)
d = torch.tensor(a)
print('init:{:}\nTensor:{:}\nfrom_numpy:{:}\ntensor:{:}'.format(a[2], b[2], c[2], d[2]))
a = np.array([1,1,1])


print('\n\ninit:{:}\nTensor:{:}\nfrom_numpy:{:}\ntensor:{:}'.format(a[2], b[2], c[2], d[2]))
"""
init:60.33664034634357
tensor:60.336639404296875
from_numpy:60.336640346343
"""