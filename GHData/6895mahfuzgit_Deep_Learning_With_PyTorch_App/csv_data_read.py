# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 16:42:48 2021

@author: E440
"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

df=pd.read_csv('arrhythmia.data',header=None)

data=df[[0,1,2,3,4,5]]
data.columns=['age','sex','height','weight','QRS duratin','P-R interval']

#plt.rcParams['figure.figsize']=[15,15]
#data.hist();

scatter_matrix(data);
