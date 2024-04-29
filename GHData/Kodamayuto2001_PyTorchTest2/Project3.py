from torchvision import transforms
import torchvision
import numpy as np
import torch
import cv2
import os

class MyDataLoader:
    count = -1
    myList = [[0]*4 for i in range(3)]
    
    def __init__(self):
        MyDataLoader.count+=1
        if MyDataLoader.count >= 4:
            MyDataLoader.count = 0
        print("コンストラクタ"+str(MyDataLoader.count)+"番目")
        pass
    def pathDir(self,m_path):
        MyDataLoader.myList[MyDataLoader.count-1] = os.listdir(m_path)
    def showPathList(self):
        print(MyDataLoader.myList[MyDataLoader.count-1])
    def readImg(self):
        MyDataLoader.myList[MyDataLoader.count-1]

x_train = MyDataLoader()
x_test = MyDataLoader()
y_train = MyDataLoader()
y_test = MyDataLoader()


x_train.pathDir(m_path="Datasets/TrainData/")
x_train.showPathList()

x_test.pathDir(m_path="Datasets/TestData/")
x_test.showPathList()

