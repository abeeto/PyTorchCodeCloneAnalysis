# -*- coding: utf-8 -*

from flyai.processor.base import Base
import os
from path import DATA_PATH
import numpy as np

'''
把样例项目中的processor.py件复制过来替换即可
'''
# 所有类别， label_list这里不需要修改
label_list = ['apron', 'bare-land', 'baseball-field', 'basketball-court', 'beach', 'bridge', 'cemetery', 'church', 'commercial-area', 'desert',
              'dry-field', 'forest', 'golf-course', 'greenhouse', 'helipad', 'ice-land', 'island', 'lake', 'meadow', 'mine',
              'mountain', 'oil-field', 'paddy-field', 'park', 'parking-lot', 'port', 'railway', 'residential-area', 'river', 'road',
              'roadside-parking-lot', 'rock-land', 'roundabout', 'runway', 'soccer-field', 'solar-power-plant', 'sparse-shrub-land', 'storage-tank', 'swimming-pool', 'tennis-court',
              'terraced-field', 'train-station', 'viaduct', 'wind-turbine', 'works']
label_num = len(label_list)


class Processor(Base):

    """
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
    该方法字段与app.yaml中的input:->columns:对应
    """

    def input_x(self, img_path):
        return img_path

    '''
    参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。
    该方法字段与app.yaml中的output:->columns:对应
    '''
    def input_y(self, label):
        return label

    '''
    参数为csv中作为输入x的一条数据，该方c法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
    该方法字段与app.yaml中的input:->columns:对应
    '''

    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''
    def output_y(self, pred_index):
        return label_list[int(pred_index)]
