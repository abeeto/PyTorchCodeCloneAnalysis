#!/usr/local/bin/python
import numpy as np
import functools as f
from mytorch.tensor import Tensor

def split(arr, indices = None, axis = 0):
    try:
        Ntotal = arr.shape[axis]
    except AttributeError:
        Ntotal = len(arr)
    # print(f'Ntotal {Ntotal} indices {indices}')
    # # l = [f.reduce(lambda x, y: x + y, j) for i in indices if i is not None for j in indices[:i] if j is not None]
    # l = [indices[:i] for i in indices]
    # # ll = [map(lambda x, y: x + y, j) for i in indices for j in indices[:i]]
    # print(f'l {l} ll {ll}')
    print(f'arr[:1] {arr[:1]}')
    i_p = 0
    ls = []
    # for i,row in enumerate(arr):
    #     print(f'i {i} row {row}')
    print(f'arr[0:indices[0]] {arr[0:indices[0]]}')
    for i,row in enumerate(arr):
        #  
        print(f'\ni {i} indices[i]')
        # ls.append(arr[i_p:indices[i]])
        # i_p = indices[i]
    # ls.append(arr[i_p:])
    

    for i in range(len(indices)):
        print(f'\ni {i} indices[i] {indices[i]} i_p {i_p} \narr[i_p:indices[i]] \n{arr[i_p:indices[i]]}')
        ls.append(arr[i_p:indices[i]])
        i_p = indices[i]

    print(f'len ls {len(ls)} \n ls \n {ls}')
    # print(f'ls[0] type {type(ls[0])} \n{ls[0]}')
    # print(f'ls[1] \n{ls[1]}')
    # print(f'ls[2] \n{ls[2]}')
    # print(f'ls[3] \n{ls[3]}')

    [print(f'ls[{i}] type {type(ls[i])} \n{ls[i]}') for i in range(len(ls))]
if __name__ == "__main__":
    
    arr = np.random.randn(4,2)
    indices = [1, 2, 3, 4]
    print(f'arr \n{arr}')

    split(arr,indices)

    arr_2 = np.random.randn(9,3)
    indices = [3,6,8,9]
    print(f'arr_2 \n{arr_2}')

    split(arr_2,indices)

    arr_3 = np.random.randn(17,3)
    indices = [4,4,3,2,2,1,1]
    i = np.array(indices)
    cu = np.cumsum(i)
    print(f'cu {cu} list cu {list(cu)}')
    print(f'arr_3 \n{arr_3}')

    split(arr_3,cu)

    arr_4 = Tensor.randn(17,3)
    indices = [4,4,3,2,2,1,1]
    i = np.array(indices)
    cu = np.cumsum(i)
    print(f'cu {cu} list cu {list(cu)}')
    print(f'arr_3 \n{arr_4}')

    split(arr_4,cu)

    ls = arr_4.split(indices)

    [print(f'ls[{i}] type {type(ls[i])} \n{ls[i]}') for i in range(len(ls))]