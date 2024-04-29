#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020-12-10 21:57
# @Author  : NingAnMe <ninganme@qq.com>
import os
from flyai.dataset import Dataset
from model import Model
from sklearn.metrics import f1_score

dataset = Dataset()
model = Model(dataset)
images_train, labels_train, images_val, labels_val = dataset.get_all_data()
path = set()
for i in images_val:
    path.add(i['img_path'])
print(len(path))
for i in images_train:
    path.add(i['img_path'])
print(len(path))


x_test, y_test = dataset.evaluate_data_no_processor('dev.csv')
print(x_test)
print(y_test)
for i1, i2 in zip(x_test, y_test):
    img_name = os.path.dirname(i1['img_path'])
    label_name = i2['label']
    if img_name != label_name:
        print(img_name, label_name, i1['img_path'])

# 用于测试 predict_all 函数
preds = model.predict_all(x_test)
labels = [i['label'] for i in y_test]

f1 = f1_score(labels, preds, average='micro')
print(f1)

for i1, i2 in zip(preds, labels):
    if i1 != i2:
        print(i1, i2)
