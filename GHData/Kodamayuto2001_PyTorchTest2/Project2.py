import torch
import torchvision
from torchvision import transforms
import numpy as np
import cv2
import os

#データの読み込み
pathX_TR = "Resources/Image/"
#pathX_TE
#pathY_TR
#pathY_TE

myList = [[0]*4 for i in range(3)]
myList[0] = os.listdir(pathX_TR)

for i in range(0,len(myList[0])):
    pathCnt = pathX_TR+myList[0][i]
    print(pathCnt)
    imgCnt = cv2.imread(pathCnt,cv2.IMREAD_UNCHANGED)
    cv2.imshow("Image"+str(i),imgCnt)

cv2.waitKey(10000)
cv2.destroyAllWindows()

#x:データ y:ラベル
#x_train,x_test,y_train,y_test

class MyDataLoader:
    def __init__(self):
        pass
    def pathDir(self,m_path):
        myList = os.listdir(m_path)
        print(myList)

x_train = MyDataLoader()
x_train.pathDir(m_path="Resources/Image/")
        
        
