import torch
from Models.YOLO.config.parser import parse_cfg, create_modules
from Models.YOLO.darknet import *
import os
from Models.Utility.ResourceProvider import *
from Models.Utility.DataLoader import test_dicom_reader
import matplotlib.pyplot as plt

#x = os.getcwd()
#x += "\\Models\\YOLO\\config\\yolov3.cfg"
#print(x)

# mass_case_description_test_set.csv
# only_MLO_set.csv
# only_CC_set.csv
y = os.getcwd()
y += "\\Data\\mass_case_description_test_set.csv"
test = ResourceProvider(y, "D:\\DataSet\\CBIS-DDSM\\", "D:\\DataSet\\ROI\\CBIS-DDSM\\")
test.read()
test.prep_grand_truth_box()
#test_dicom_reader()
print("skończyłem")
#YOLO = YOLODeepNet(x)
#YOLO.forward(0, True)

