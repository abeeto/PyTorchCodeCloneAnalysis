
# -*- coding: utf-8 -*-
import copy
import csv
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset_path = './Data.csv'

def load_dataset(path=None):
    if path is None:
        path = dataset_path
    data = pd.read_csv(path)  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
    new_data = data.as_matrix()
    return new_data

if __name__ == '__main__':
    load_dataset()
