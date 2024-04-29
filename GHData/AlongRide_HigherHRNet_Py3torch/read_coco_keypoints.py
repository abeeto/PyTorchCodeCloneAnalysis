
# 提取COCO关键点并保存在CSV文件中 Date: 2018.3.22

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
from PIL import Image
from PIL import ImageDraw
import csv
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# initialize COCO api for person keypoints annotations
dataDir='/home/cidi/py3torch/HigherHRNet/data/coco'
dataType='train2017'
annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
coco_kps=COCO(annFile)

# display COCO categories and supercategories
cats = coco_kps.loadCats(coco_kps.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
catIds = coco_kps.getCatIds(catNms=['person']);
imgIds = coco_kps.getImgIds(catIds=catIds );
print ('there are %d images containing human'%len(imgIds))

def getBndboxKeypointsGT():
    csvFile = open('KeypointBndboxGT.csv','w') 
    keypointsWriter = csv.writer(csvFile)
    firstRow = ['imageName','personNumber','bndbox','nose',
            'left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
            'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip',
            'left_knee','right_knee','left_ankle','right_ankle']
    keypointsWriter.writerow(firstRow)
    for i in range(len(imgIds)):
        imageNameTemp = coco_kps.loadImgs(imgIds[i])[0]
        imageName = imageNameTemp['file_name'].encode('raw_unicode_escape')
        img = coco_kps.loadImgs(imgIds[i])[0]
        annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco_kps.loadAnns(annIds)
        personNumber = len(anns)
        for j in range(personNumber):
            bndbox = anns[j]['bbox']
            keyPoints = anns[j]['keypoints']
            keypointsRow = [imageName,str(personNumber),
                            str(bndbox[0])+'_'+str(bndbox[1])+'_'+str(bndbox[2])+'_'+str(bndbox[3]),
                            str(keyPoints[0])+'_'+str(keyPoints[1])+'_'+str(keyPoints[2]),
                            str(keyPoints[3])+'_'+str(keyPoints[4])+'_'+str(keyPoints[5]),
                            str(keyPoints[6])+'_'+str(keyPoints[7])+'_'+str(keyPoints[8]),
                            str(keyPoints[9])+'_'+str(keyPoints[10])+'_'+str(keyPoints[11]),
                            str(keyPoints[12])+'_'+str(keyPoints[13])+'_'+str(keyPoints[14]),
                            str(keyPoints[15])+'_'+str(keyPoints[16])+'_'+str(keyPoints[17]),
                            str(keyPoints[18])+'_'+str(keyPoints[19])+'_'+str(keyPoints[20]),
                            str(keyPoints[21])+'_'+str(keyPoints[22])+'_'+str(keyPoints[23]),
                            str(keyPoints[24])+'_'+str(keyPoints[25])+'_'+str(keyPoints[26]),
                            str(keyPoints[27])+'_'+str(keyPoints[28])+'_'+str(keyPoints[29]),
                            str(keyPoints[30])+'_'+str(keyPoints[31])+'_'+str(keyPoints[32]),
                            str(keyPoints[33])+'_'+str(keyPoints[34])+'_'+str(keyPoints[35]),
                            str(keyPoints[36])+'_'+str(keyPoints[37])+'_'+str(keyPoints[38]),
                            str(keyPoints[39])+'_'+str(keyPoints[40])+'_'+str(keyPoints[41]),
                            str(keyPoints[42])+'_'+str(keyPoints[43])+'_'+str(keyPoints[44]),
                            str(keyPoints[45])+'_'+str(keyPoints[46])+'_'+str(keyPoints[47]),
                            str(keyPoints[48])+'_'+str(keyPoints[49])+'_'+str(keyPoints[50]),]

            keypointsWriter.writerow(keypointsRow)

    csvFile.close()

if __name__ == "__main__":
    print ('Writing bndbox and keypoints to csv files..."')
    getBndboxKeypointsGT() 
