import os
import subprocess
import re
sysMsg = subprocess.getstatusoutput('ifconfig')
result = re.findall( r'[0-9]+(?:\.[0-9]+){3}', sysMsg[1] )


import socket
import time
from imutils.video import VideoStream
import imagezmq

sender = imagezmq.ImageSender(connect_to='tcp://192.168.43.96:5555')

rpi_name = socket.gethostname() # send RPi hostname with each image

picam = VideoStream(usePiCamera=True).start()
time.sleep(2.0)  # allow camera sensor to warm up

while True:  # send images as stream until Ctrl-C
  image = picam.read()
  sender.send_image(rpi_name, image)
