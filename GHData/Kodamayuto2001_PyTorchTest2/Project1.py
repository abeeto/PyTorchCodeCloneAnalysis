import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import os

#データ読み込み
pathDir = "Resources/Image/"

myDataset = []

#ファイルとディレクトリの一覧を取得
myList = os.listdir(pathDir)
print(myList)

for i in range(0,len(myList)):
    pathCnt = pathDir+myList[i]
    print(pathCnt)
    imgCnt = cv2.imread(pathCnt,cv2.IMREAD_UNCHANGED)
    print(len(imgCnt))
    print(type(imgCnt))
    cv2.imshow("Image"+str(i),imgCnt)
    imgTensor = torch.from_numpy(imgCnt).clone()
    print(type(imgTensor))
    #imgTensor.shape()
    myDataset.append(imgTensor)

cv2.waitKey(10000)
cv2.destroyAllWindows()

#データをロード


torch.utils.data.DataLoader(dataset=myDataset[0],batch_size=10,shuffle=True)

