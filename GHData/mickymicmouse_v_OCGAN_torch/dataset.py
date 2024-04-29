import os
import sys
import torchvision.transforms as transforms
import numpy as np
import options
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
import time
import options
import random
import tqdm
import shutil


def dataspread():
    
    opt=options.train_options()
    normal = str(opt.normal_class)
    normal = normal.replace(" ","")
    normal = normal.split(",")
    dataroot = opt.dataroot
    datasource = opt.datasource
    print("Spread dataset")
    for i in normal:
        if not i in os.listdir(datasource):
            print("no class in data source")
            return 0
    abnormal =str(opt.abnormal_class)
    abnormal = abnormal.replace(" ","")
    abnormal = abnormal.split(",")
    for i in abnormal:
        if not i in os.listdir(datasource):
            print("no class in data source")
            return 0

    if os.path.isdir(dataroot):
        shutil.rmtree(dataroot, ignore_errors = True)
    if not os.path.exists(dataroot):
        os.mkdir(dataroot)
    if not os.path.exists(os.path.join(dataroot,"training")):
        os.mkdir(os.path.join(dataroot, "training"))
    if not os.path.exists(os.path.join(dataroot,"validation")):
        os.mkdir(os.path.join(dataroot, "validation"))
    if not os.path.exists(os.path.join(dataroot, "testing")):    
        os.mkdir(os.path.join(dataroot, "testing"))
    if not os.path.exists(os.path.join(dataroot, "validation","1")):    
        os.mkdir(os.path.join(dataroot, "validation","1"))
    if not os.path.exists(os.path.join(dataroot,"training","1")):
        os.mkdir(os.path.join(dataroot, "training","1"))
    if not os.path.exists(os.path.join(dataroot,"testing","1")):
        os.mkdir(os.path.join(dataroot,"testing","1"))
    if not os.path.exists(os.path.join(dataroot,"testing","0")):    
        os.mkdir(os.path.join(dataroot,"testing", "0"))
        
    train_root = os.path.join(dataroot, "training")
    valid_root = os.path.join(dataroot, "validation")
    test_root = os.path.join(dataroot, "testing")
    abs_normal_list=[]
    for i in normal:
        normal_list = []
        normal_list = os.listdir(os.path.join(datasource, i))
        for j in normal_list:
            abs_normal_list.append(os.path.join(datasource,i,j))
    random.shuffle(abs_normal_list)
    tv_list = abs_normal_list[:int(len(abs_normal_list)*0.8)]
    valid_list = tv_list[:int(len(tv_list)*0.1)]
    train_list = tv_list[int(len(tv_list)*0.1):]
    test_list = abs_normal_list[int(len(abs_normal_list)*0.8):]
    nrm=0
        #train
    for i in train_list:
        file_name = str(nrm)+".png"
        src = i
        dst = os.path.join(train_root, "1", file_name)
        nrm=nrm+1
            
        shutil.copy(src, dst)
        #valid
    for i in valid_list:
        file_name = str(nrm)+".png"
        src = i
        dst = os.path.join(valid_root, "1", file_name)
        nrm=nrm+1
            
        shutil.copy(src, dst)
        #test_normal
    for i in test_list:
        file_name = str(nrm)+".png"
        src = i
        dst = os.path.join(test_root, "1", file_name)
        nrm=nrm+1
            
        shutil.copy(src, dst)
    #test_abnormal
    nrm_test = len(test_list)
    ab_nrm_each = int(nrm_test//len(abnormal))

    for i in abnormal:
        files = os.listdir(os.path.join(datasource,i))
        random.shuffle(files)
        files=files[:ab_nrm_each]
        for j in files:
            file_name = str(nrm)+".png"
            src = os.path.join(datasource, i, j)
            dst = os.path.join(test_root, "0", file_name)
            nrm=nrm+1
            shutil.copy(src, dst)
    print("complete data split")


def dataspread2():
    
    opt=options.train_options()
    normal = str(opt.normal_class)
    normal = normal.replace(" ","")
    normal = normal.split(",")
    dataroot = opt.dataroot
    train_datasource = os.path.join(opt.datasource,"training")
    test_datasource = os.path.join(opt.datasource, "testing")
    print("Spread dataset")
    for i in normal:
        if not i in os.listdir(train_datasource):
            print("no class in data source")
            return 0
    abnormal =str(opt.abnormal_class)
    abnormal = abnormal.replace(" ","")
    abnormal = abnormal.split(",")
    for i in abnormal:
        if not i in os.listdir(train_datasource):
            print("no class in data source")
            return 0

    if os.path.isdir(dataroot):
        shutil.rmtree(dataroot, ignore_errors = True)
    if not os.path.exists(dataroot):
        os.mkdir(dataroot)
    if not os.path.exists(os.path.join(dataroot,"training")):
        os.mkdir(os.path.join(dataroot, "training"))
    if not os.path.exists(os.path.join(dataroot,"validation")):
        os.mkdir(os.path.join(dataroot, "validation"))
    if not os.path.exists(os.path.join(dataroot, "testing")):    
        os.mkdir(os.path.join(dataroot, "testing"))
    if not os.path.exists(os.path.join(dataroot, "validation","1")):    
        os.mkdir(os.path.join(dataroot, "validation","1"))
    if not os.path.exists(os.path.join(dataroot,"training","1")):
        os.mkdir(os.path.join(dataroot, "training","1"))
    if not os.path.exists(os.path.join(dataroot,"testing","1")):
        os.mkdir(os.path.join(dataroot,"testing","1"))
    if not os.path.exists(os.path.join(dataroot,"testing","0")):    
        os.mkdir(os.path.join(dataroot,"testing", "0"))
        
    train_root = os.path.join(dataroot, "training")
    valid_root = os.path.join(dataroot, "validation")
    test_root = os.path.join(dataroot, "testing")
    abs_normal_list=[]
    for i in normal:
        normal_list = []
        normal_list = os.listdir(os.path.join(train_datasource, i))
        for j in normal_list:
            abs_normal_list.append(os.path.join(train_datasource,i,j))
    random.shuffle(abs_normal_list)
    train_list = abs_normal_list[:int(len(abs_normal_list)*0.9)]
    valid_list = abs_normal_list[int(len(abs_normal_list)*0.9):]

    nrm = 0
        #train
    for i in train_list:
        file_name = str(nrm)+".png"
        src = i
        dst = os.path.join(train_root, "1",file_name)
        nrm=nrm+1
            
        shutil.copy(src, dst)
        #valid
    for i in valid_list:
        file_name = str(nrm)+".png"
        src = i
        dst = os.path.join(valid_root, "1")
        nrm=nrm+1
            
        shutil.copy(src, dst)
        #test_normal

    for i in normal:
        files = os.listdir(os.path.join(test_datasource,i))
        random.shuffle(files)
        for j in files:
            file_name = str(nrm)+".png"
            src = os.path.join(test_datasource, i, j)
            dst = os.path.join(test_root, "1",file_name)
            nrm=nrm+1
            shutil.copy(src, dst)

    #test_abnormal
    for i in abnormal:
        files = os.listdir(os.path.join(test_datasource,i))
        random.shuffle(files)
        for j in files:
            file_name = str(nrm)+".png"
            src = os.path.join(test_datasource, i, j)
            dst = os.path.join(test_root, "0", file_name)
            nrm=nrm+1
            shutil.copy(src, dst)
    print("complete data split")

        


