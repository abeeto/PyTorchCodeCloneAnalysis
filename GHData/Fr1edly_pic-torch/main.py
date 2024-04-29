from cv2 import FONT_HERSHEY_COMPLEX, FONT_HERSHEY_COMPLEX_SMALL
from matplotlib import transforms
from torch.serialization import load
import scipy.misc as misc
import time
import model as m
import numpy as np
#import train
import cv2 as cv
import data as d
import torch
import torchvision.utils as utils
import csv, io
from PIL import Image

def run():
    torch.multiprocessing.freeze_support()
    
if __name__ == "__main__":
    run()
    #dataiter = iter(torch.utils.data.DataLoader(d.test_loader, batch_size=batch_size, shuffle=True) ) 
    dataiter = iter(d.test_loader)

    net = m.imgNet()
    
    net.load_state_dict(torch.load('model.pth'))
    net.eval()
    path = 'assets\images\signs.csv'
    class_text = io.open(path, encoding='utf-8')
    csv_classes = csv.reader(class_text)
    classes = [f[0].split(';') for f in csv_classes]

    #cam = cv.VideoCapture(1)
    #cam.set(cv.CAP_PROP_FPS, 60)
    while True:
        """    
        _, frame = cam.read()
        img = cv.resize(frame, (32,32))
        #frame = frame.float()
        img = d.transform(img)
        img = img.unsqueeze(0)
        """
        images, labels = dataiter.next()
        
        qwe = utils.make_grid(images)
        qwe = qwe/2+0.5
        npimg = qwe.numpy()
        npimg= np.transpose(npimg,(1,2,0))
    
        #npimg = cv.resize(npimg, (96*d.batch_size,96))
        
        """image = PIL.Image.fr omarray(frame)
        image.thumbnail((32,32))
        image = d.transform(image)
        image = image.float()
        image = image.cpu()
        image = image.unsqueeze(0)"""
        #print('GroundTruth: ', ' '.join('%5s' % d.classes[labels[j]] for j in range(batch_size)))

        outputs = net(images)

        _, predic = torch.max(outputs, 1)
        #print(predic, end='\r')
        print('Predicted: ', ' '.join('%5s ;' % d.classes[predic[j]] for j in range(d.batch_size)),end='\r' )
        #npimg = np.ascontiguousarray(npimg, dtype=np.uint8)
       # for w in range(8):
       #     for h in range(2):
       #         for i in range(d.batch_size):
       #             cv.putText(img = npimg, text=d.classes[predic[i]], org=(25*w,35*h), color=(0,255,0), fontScale=1, fontFace=cv.FONT_HERSHEY_COMPLEX_SMALL)
        cv.imshow(f'123',npimg)

        cv.waitKey(5000)