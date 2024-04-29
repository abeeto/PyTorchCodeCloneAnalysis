import socket
import os
import subprocess
from subprocess import Popen, PIPE, STDOUT
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(4,GPIO.IN)

while True:
	if GPIO.input(4)== 1:
		#empty quote
		open('clay1.txt', 'w').close()
		# timer triggers webcam to take image
		os.system('fswebcam --no-banner -S 20 claycam.jpg')
		# send image
		s = socket.socket()
		host = "147.75.102.59"
		port = 12345
		s.connect((host, port))
		time.sleep (2)
		f=open ("claycam.jpg", "rb") 
		l = f.read(4096)
		while (l):
      			s.send(l)
     			l = f.read(4096)
		time.sleep(3)
		print "image sent"   
		s.close()                # Close the connection

    		# wait for densecap & torch-rnn to write quote
		print "done"
		time.sleep(6)
		s = socket.socket()
		host = "147.75.102.59"
		port = 12345
                
		s.connect((host, port))
		l = s.recv(1024)
		clay = open("clay1.txt", "w")
		clay.write(l)
		clay.close()			
		s.send("file received")
		s.close()
		
