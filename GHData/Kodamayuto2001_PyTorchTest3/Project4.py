#region IMPORT
from __future__ import division, print_function
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from skimage import io, transform
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import torchvision
import pprint
import torch
import cv2
import os
#endregion

#このファイルと同じディレクトリにhaarlike特徴分類器をダウンロードしてください。
#https://github.com/opencv/opencvにアクセス

class MyDataset:
    def __init__(self,pathImgTrim,tagName):
        self.__setPath(pathImgTrim)
        self.__setTag(tagName)
        self.__ndarrayTransform()
        self.__CSV_transform()
        self.__setImgSize()
        self.__Rescale()
        print("-----処理終了-----")

    def MyMakeData(self):
        #face_cascade = cv2.CascadeClassifier('opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
        cascade_path = "opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path) # detectorを作成

        cap = cv2.VideoCapture(0)
        cap.set(3,340)
        cap.set(4,200)

        cnt = 0

        while True:
            success,img = cap.read()
            pathImage = "Resources/Images/"+str(cnt)+".jpg"
            pathImgTrim = "Resources/ImgTrim/"+str(cnt)+".jpg"

            imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            imgResults = img.copy()

            #カスケード分類器の特徴量を取得する
            cascade = cv2.CascadeClassifier(cascade_path)
            facerect = cascade.detectMultiScale(imgGray,scaleFactor=1.1,minNeighbors=2,minSize=(10,10))

            color = (82,188,222)

            

            #検出した場合
            if len(facerect) > 0:
                for rect in facerect:
                    cv2.rectangle(imgResults,tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
                    x = rect[0]
                    y = rect[2]
                    xw,yh = tuple(rect[0:2]+rect[2:4])
                    imgTrim = img[y:yh, x:xw]

                    #保存
                    cv2.imwrite(pathImgTrim,imgTrim)
                    cv2.imwrite(pathImage,img)

                    #表示
                    cv2.imshow("Results",imgResults)
                    cv2.imshow("ImageTrim",imgTrim)


            cv2.imshow("Original",img)

            if cnt >= 99:
                cnt = 0
            else:
                cnt += 1

            if cv2.waitKey(100) & 0xFF == ord("q"):
                break

    def __setImgSize(self):
        #すべての画像のサイズ(高さと幅)の平均値(int)
        #高さ
        self.__defH = 0
        self.__defW = 0
        self.__defC = 0
        for i in range(0,len(self.__myFileName)):
            self.__defH += self.__h[i]
            self.__defW += self.__w[i]
            self.__defC += self.__c[i]
        self.__defH /= len(self.__h)
        self.__defW /= len(self.__w)
        self.__defC /= len(self.__c)
        self.__defH = int(self.__defH)
        self.__defW = int(self.__defW)
        self.__defC = int(self.__defC)

    def __setPath(self,pathImgTrim):
        self.__pathImgTrim = pathImgTrim
        self.__myFileName = os.listdir(self.__pathImgTrim)
        #print(type(self.__myFileName))
        #print(len(self.__myFileName))

    def __setTag(self,tagName="Kodama"):
        self.__tag = tagName

    def __numpyTransform(self):
        self.__numpyImg = []
        self.__myFilePath = []
        for i in range(0,len(self.__myFileName)):
            self.__myFilePath.append(self.__pathImgTrim+str(self.__myFileName[i]))
            self

    def __ndarrayTransform(self):
        self.__myFilePath = []
        self.__ndArrayImg = []
        self.__h = []
        self.__w = []
        self.__c = []
        for i in range(0,len(self.__myFileName)):
            self.__myFilePath.append(self.__pathImgTrim+str(self.__myFileName[i]))
            self.__ndArrayImg.append(np.array(Image.open(self.__myFilePath[i]),np.float64))
            self.__h.append(self.__ndArrayImg[i].shape[0])
            self.__w.append(self.__ndArrayImg[i].shape[1])
            self.__c.append(self.__ndArrayImg[i].shape[2])
            
    def __CSV_transform(self):
        self.__cnt = 0
        self.__ListFN = []
        self.__ListTag = []
        while self.__cnt < len(self.__myFileName):
            #print(self.__myFileName[self.__cnt])
            self.__ListFN.append(self.__myFileName[self.__cnt])
            self.__ListTag.append(self.__tag)

            self.__cnt += 1
        #print(self.__ListFN)
        #print(self.__ListTag)
        self.__CSV_Data = pd.DataFrame(data=self.__ListTag,index=self.__ListFN)
        self.__osPathCSV_1 = "Data/"
        try:
            os.makedirs(self.__osPathCSV_1)
        except FileExistsError:
            pass
        
        self.__pathCSV = self.__osPathCSV_1+"data_"+str(self.__tag)+".csv"
        self.__CSV_Data.to_csv(self.__pathCSV)
    
    def __Rescale(self):
        self.__resImg = []
        for i in range(0,len(self.__myFileName)):
            self.__resImg.append(cv2.resize(self.__ndArrayImg[i],dsize=(self.__defW,self.__defH)))
            pass
        pass

    def showFileName(self):
        print(self.__myFileName)
    def showCSV_File(self):
        print(self.__CSV_Data)
    def showDefSize(self):
        print("平均値h：{}".format(self.__defH))
        print("平均値w：{}".format(self.__defW))
        print("平均値c：{}".format(self.__defC))
    def showNdArrayTF(self):
        for i in range(0,len(self.__myFileName)):
            print(type(self.__ndArrayImg[i]))
            """
            print(self.__ndArrayImg[i].shape)
            print("height   :{}".format(self.__ndArrayImg[i].shape[0]))
            print("width    :{}".format(self.__ndArrayImg[i].shape[1]))
            print("channel  :{}".format(self.__ndArrayImg[i].shape[2]))
            """
            print("------------------------------------------")
            #print(self.__ndArrayImg[i])
            print("height   :{}".format(self.__h[i]))
            #print("height   :\n{}".format(self.__ndArrayImg[i][0]))
            print("width    :{}".format(self.__w[i]))
            #print("width    :\n{}".format(self.__ndArrayImg[i][1]))
            print("channel  :{}".format(self.__c[i]))
            #print("channel  :\n{}".format(self.__ndArrayImg[i][2]))
    def showRescale(self):
        for i in range(0,len(self.__myFileName)):
            print("h    :{}".format(self.__resImg[i].shape[0]))
            print("w    :{}".format(self.__resImg[i].shape[1]))
            print("c    :{}".format(self.__resImg[i].shape[2]))
    def showTest(self):
        #print(self.__ndArrayImg)
        #print(type(self.__ndArrayImg))
        #print(len(self.__h))
        #print(len(self.__w))
        #print(len(self.__c))
        print(type(self.__ndArrayImg[0]))
        print(len(self.__myFileName))
        cv2.imshow()
        pass       


x = MyDataset(pathImgTrim="Resources/ImgTrim/",tagName="Kodama")
#データがないときはMyDatasetクラスのMyMakeData関数を呼んでデータを作ってください。
x.MyMakeData()
#x.showFileName()
#x.showCSV_File()
#x.showDefSize()
#x.showNdArrayTF()
#x.showRescale()
