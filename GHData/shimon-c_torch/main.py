# This is a sample Python script.
import numpy as np
import torch
import imageio
import cv2
from torch import nn as nn

# to install with GPU
# pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

#days_arr = np.loadtxt('C:/Users/shimon.cohen/data/bikes/day.csv',
#                      dtype=np.float32,
#                      delimiter=",",
#                      skiprows=1,
#                      converters={1, lambda x: float(x[8:10])}
#                      )
#hours_arr = np.loadtxt('C:/Users/shimon.cohen/data/bikes/hour.csv',
#                       dtype=np.float32,
#                       delimiter=",",
#                       skiprows=1,
#                       converters={1, lambda x: float(x[8:10])}
#                       )
#days_arr.shape
#hours_arr.shape
one_hot = np.arange(5)
arr = torch.zeros(5,5)
arr1 = np.arange(20)
tarr = torch.tensor(arr1, dtype=torch.float32)
tarr = torch.reshape(tarr, (2,10))
tarr
net = nn.ELU()
def norm_image(img):
    C = img.shape[0]
    EPS = 1e-8
    for c in range(C):
        mn = torch.mean(img[c,:])
        ss = torch.std(img[c,:], unbiased=True) + EPS
        img[c,:] = (img[c,:] - mn)/ss
    return img

def norm_data(data):
    if len(data.shape)>3:
        N = data.shape[0]
        for n in range(N):
            data[n,:] = norm_image((data[n,:]))
    else:
        data = norm_image(data)
    return data

class Scope:
    i = 0
    def __enter__(self):
        print('enter')
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('exit')

    def __next__(self):
        self.i += 1
        return self.i
    def __iter__(self):
        return self

# some tests of python
lst = [x for x in range(50)]
l1 = lst[:-10]
l2 = lst[-10:]
print(f'l1={l1}')
print(f'l2={l2}')
doit = True
with l1 in Scope() as s:
    l1r = l1[-1::-1]
    print(f'l1r={l1r}')
def mul(x1 : float,x2 : float) -> float:
    y = x1*x2
    return y

pp = [3,5]
print(f'mul(,3,5) = {mul(*pp)}')

img_path = 'C:/Users/shimon.cohen/data/OrSenese/test_dir/data_dir/Good/Good1.PNG'
img = imageio.read(img_path)
#timg = torch.from_numpy(img)
cimg = cv2.imread(img_path)
timg = torch.from_numpy(cimg)
timg = timg.permute([2,0,1])
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
