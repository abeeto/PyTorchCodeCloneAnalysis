# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 22:38:50 2020

@author: cj
"""

import cv2
import imagezmq
from time import gmtime, strftime
 
image_hub = imagezmq.ImageHub()

while True:
  rpi_name, image = image_hub.recv_image()

  image2= cv2.resize(image, dsize=(640*5, 480*5), interpolation=cv2.INTER_AREA)  
  cv2.imshow(rpi_name, image2)

  imgfile='D:/image/'+strftime("%Y%m%d_%H_%M_%S", gmtime())+'.png'
  cv2.imwrite(imgfile, image2)

#  if cv2.waitKey(1) == ord('q'):
#    break
  
  image_hub.send_reply(b'OK')


