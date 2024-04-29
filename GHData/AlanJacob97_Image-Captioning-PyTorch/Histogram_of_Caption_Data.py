#!/usr/bin/env python
# coding: utf-8

# In[2]:


import dataset as ds
#import model as mdl

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as vis
#from torchtext import data
import numpy as np

from collections import OrderedDict as OD
from copy import copy


# In[3]:


#dir_photos = "./data/flickr8k/Flicker8k_photos/"
#file_annot = "./data/flickr8k/Flickr8k_text/Flickr8k.token.txt"
#jpg_files = ds.images_info(dir_photos)
ann_dframe = ds.annots_info('Flickr8k.token.txt', df=True)
# print(ann_dframe)

## Prepare captions
word_count = ds.word_freq(ann_dframe)


# In[8]:


z=word_count.loc[1:20, ['word', 'count']] 
z


# In[9]:


word_freq_dict = dict(zip(z['word'], z['count']))
word_freq_dict


# import matplotlib.pyplot as plt
# fig= plt.figure(figsize=(10,10))
# plt.barh(list(word_freq_dict.keys()), word_freq_dict.values(), color='g')
# plt.xlabel('Word frequency')
# plt.ylabel('Words')

# In[4]:


# ## Clean text
    # # print(ann_dframe.caption.values)
    # print("Cleaning text ... ", end="")
for i, cpt in enumerate(ann_dframe.caption.values):
    ann_dframe["caption"].iloc[i] = ds.clean_text(cpt)
print("done.")
    # # print(ann_dframe)
word_count = ds.word_freq(ann_dframe)


# In[5]:


z=word_count.loc[1:20, ['word', 'count']] 
z


# In[6]:


word_freq_dict = dict(zip(z['word'], z['count']))
word_freq_dict


# In[7]:


import matplotlib.pyplot as plt
fig= plt.figure(figsize=(7,7))
plt.barh(list(word_freq_dict.keys()), word_freq_dict.values(), color='g')
plt.xlabel('Word frequency')
plt.ylabel('Words')
plt.title('The top 20 most frequently occuring words after cleaning data')


# In[ ]:




