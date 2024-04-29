# coding:utf-8

from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time
from faceframework import face_model
from datetime import timedelta

from PIL import Image, ImageDraw
from multiprocessing import Process,Manager
from svccontrol import face_lib_manager
from faceframework.utils import faceutils
import numpy as np
from faceframework.align.api_mtcnn import MTCNN
from faceframework.align.api_centerface import CenterFaceAPI

import facesvc
facemodel = None
facelib = None


def get_imgface(detector, img):
    """
    zhujinhua 20200602
    Arguments:
        img: an instance of PIL.Image.
        min_face_size: a float number.
    Returns:
        faces:
        bboxes:
        facecount
    """
    start=time.time()
    bboxes, faces = detector.align_multi(img, 50, 30)
    if len(bboxes) == 0:
        return [],[],0
    bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
    bboxes = bboxes.astype(int)
    offsetpixes = 10
    bboxes = bboxes + [-offsetpixes,-offsetpixes,offsetpixes,offsetpixes] # personal choice
    bboxes = [[x[0] if x[0]>=0 else 0,x[1] if x[1]>=0 else 0,x[2],x[3]] for x in bboxes]
    print('get_imgface:',time.time()-start)
    return faces,bboxes,len(bboxes)

def test_detect_cv2(detector,imgfile,alg='mtcnn'):

    rawimg = cv2.imread(imgfile)

    scaleflag=False
    '''if min(rawimg.shape[1], rawimg.shape[0]) > 1000:
        size = (int(rawimg.shape[1] * 0.5), int(rawimg.shape[0] * 0.5))
        img = cv2.resize(rawimg, size)
    else:
        img = rawimg
        '''

    faces, bb_box, face_num = get_imgface(detector, rawimg)
    if faces == None or len(faces)==0:
        return []

    i=0
    for face in faces:
        face.save('data/people/'+alg+'/'+imgfile.split('/')[-1][:-4]+' '+str(i)+'.jpg')
        i=i+1
    #test
    #cv2.imwrite('data/test/'+faceutils.get_time(True)+'raw.jpg', rawimg)
    color=(255,0,0)
    if alg=='centerface':
        color=(0,255,0)
    for bbox in bb_box:
        cv2.rectangle(rawimg,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color,3)
        #cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color,3)
    cv2.imwrite(imgfile[:-4]+'_%s_%d.jpg'%(alg,face_num), rawimg)
    return faces


def facedetect():
    print('hello')
    facemodel= face_model.FaceModel('')

    folderpath='data/people/'
    mtcnn = MTCNN()
    centerface = CenterFaceAPI()

    filelist=os.listdir(folderpath)

    '''detector= mtcnn
    for imgfile in filelist:
        if not imgfile.endswith('.jpg') or len(imgfile)!=len('20200619113451.jpg'):
            continue
        test_detect_cv2(detector, folderpath+imgfile, 'mtcnn')'''

    detector= centerface
    for imgfile in filelist:
        if not imgfile.endswith('.jpg') or len(imgfile)!=len('20200619113451.jpg'):
            continue
        centerface = CenterFaceAPI()
        facescenterface=test_detect_cv2(centerface, folderpath+imgfile, 'centerface')
        facesmtcnn=test_detect_cv2(mtcnn, folderpath+imgfile, 'mtcnn')
        if len(facescenterface)==0 or len(facesmtcnn)==0:
            continue
        featcf=facemodel.get_batchfeature(facescenterface)
        featmtcnn=facemodel.get_batchfeature(facesmtcnn)
        val=np.matmul(featcf, featmtcnn.T)
        rstimg=[]
        for i in range(len(facescenterface)):
            simmax=np.argmax(val,axis=1)
            print(i,'similarity:',val[i,simmax[i]])
            img= np.concatenate([facescenterface[i], facesmtcnn[simmax[i]]], axis=1)
            if len(rstimg) == 0:
                rstimg = img
            else:
                rstimg = np.vstack((rstimg,img))
        print(val)
        rstimg = cv2.cvtColor(rstimg,cv2.COLOR_BGR2RGB)
        cv2.imwrite(folderpath+imgfile[:-4]+'merge.jpg',rstimg)
        cv2.imshow('a',rstimg)
        cv2.waitKey()








if __name__ == '__main__':
    print('11')
    facedetect()


