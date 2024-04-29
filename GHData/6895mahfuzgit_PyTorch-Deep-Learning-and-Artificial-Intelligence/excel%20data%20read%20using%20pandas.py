# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:46:00 2022

@author: Mahfuz Shazol
"""

import pandas as pd
import matplotlib.pyplot as plt
from  pandas.plotting  import scatter_matrix
#https://archive.ics.uci.edu/ml/machine-learning-databases/arrhymia/arrhymia.data

df=pd.read_csv("Datas/arrhythmia.data",header=None)
data=df[[0,1,2,3,4,5]]
data.columns=['age','sex','height','weight','QRS duration','P-R interval']
print(data)

plt.rcParams['figure.figsize']=[15,15]
data.hist()

scatter_matrix(data)






