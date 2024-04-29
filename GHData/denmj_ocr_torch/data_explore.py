import numpy as np
import pandas as pd
import copy
import csv
import functools
import glob
import os
from collections import namedtuple
from utils.logconfig import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

candidates_datav2 = pd.read_csv('D://LUNA_Dataset//candidates_V2.csv')
# candidates_data = pd.read_csv('/content/candidates.csv')
# candidates_datav2 = pd.read_csv('/content/candidates_V2.csv')
annotation_data = pd.read_csv('D://LUNA_Dataset//data_unversioned//part2//luna//annotations.csv')


CandidateInfoTuple = namedtuple('CandidateInfoTuple',
                                'isNodule_bool, diameter, series_uid, center_xyz'
                                )

# print(candidates_datav2.info())


@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):

    # Reading all mhd uids from luna sets
    mhd_list = glob.glob('D:/LUNA_Dataset/data_unversioned/part2/luna/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    # Reading uid from annotation file
    diameter_dict = {}
    for _, row in pd.read_csv('/content/annotations.csv').iterrows():
        series_uid = row[0]
        annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
        annotationDiameter_mm = float(row[4])
        diameter_dict.setdefault(series_uid, []).append((annotationCenter_xyz, annotationDiameter_mm))

    # Reading candidate_v2 file




# diamater_dict = {}
# with open('D:/LUNA_Dataset/annotations.csv', "r") as f:









