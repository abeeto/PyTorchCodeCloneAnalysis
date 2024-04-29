import os
import sys

dir_list = os.listdir('checkpoint')
dir_list = sorted(dir_list,  key=lambda x: os.path.getmtime(os.path.join('checkpoint', x)))
print(dir_list)
