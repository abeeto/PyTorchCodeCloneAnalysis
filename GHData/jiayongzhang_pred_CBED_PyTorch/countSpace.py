# -*- coding: utf-8 -*-
"""pred_CBED.ipynb

get mean and std for the whole data

Automatically generated by Colaboratory.
Referred to Pytorch ZetoToAll
Original file is located at
    https://colab.research.google.com/drive/1so8i-5nk4xpYEuXOXbHDgkIfe8b1LLav
"""
import sys
import glob
import pandas as pd
import numpy as np
import collections
from matplotlib import pyplot as plt
import h5py

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



#train_paths = glob.glob('./train/batch_train_0.h5')
train_paths = glob.glob('./' + 'train/' + '*.h5')
spaces = []
for this_train in train_paths:
    print("Current file: %s"%this_train)
    try:
        f = h5py.File(this_train, mode='r', swmr=True)
        for key in (f.keys()):
            this_space = int(f[key].attrs['space_group'])
            spaces.append(this_space)
    except:
        #warnings.warn("RuntimeError", RuntimeWarning)
        print("Warning: error handling %s, will be ignored."%this_train)
        continue


space_count = collections.Counter(spaces)
#print(space_count)
#labels, values = zip(*space_count.items())

labels, values = np.arange(230)+1, np.zeros(230)
for key,value in space_count.items():
    values[key-1] = value

#save data
'''
import pickle
pickle_file = './space_count.pkl'
try:
    with open(pickle_file, 'wb') as f:
        save = {'space_group': labels, 'count': values}
        pickle.dump(save, f, protocol=pickle.HIGHEST_PROTOCOL)
except:
    raise
    '''
space_data =  pd.DataFrame(list(zip(labels, values)), columns=["space_group", "count"])

total_count = space_data.sum()['count']
weight = space_data['count']/total_count
space_data['weight'] = weight
space_data.to_csv("count.csv",index=False)

exit()


indexes = np.arange(len(labels))
width = 1

#f, ax = plt.subplots(figsize=(100,5))
plt.figure(figsize=(115,80))
plt.bar(indexes, values, width)
#plt.xticks(indexes + width * 0.5, labels)
plt.xticks(indexes, labels)
plt.savefig("distribution.png")
plt.show()



#save data with count<100
labels, values = [], []
for label in sorted(space_count.keys()):
    if space_count[label] < 100:
        labels.append(label+1)
        values.append(space_count[label])

indexes = np.arange(len(labels))
width = 1
#f, ax = plt.subplots(figsize=(100,5))
plt.figure(figsize=(115,80))
plt.bar(indexes, values, width)
#plt.xticks(indexes + width * 0.5, labels)
plt.xticks(indexes, labels)
plt.savefig("distribution_100_lower.png")
plt.show()



f = h5py.File(sys.argv[1], mode='r', swmr=True)
keys = list(f.keys())
_len = len(keys)
tmp_x, tmp_y = [], []
for i in range(1):
    tmp_x.append(dict(f[keys[i]].items())['cbed_stack'][()])
    #tmp_y.append(dict(f[keys[i]].items())['cbed_stack'][()][1])
    #tmp_z.append(dict(f[keys[i]].items())['cbed_stack'][()][2])
    tmp_y.append(int(dict(f[keys[i]].attrs.items())['space_group']))

print(tmp_x, tmp_y)
