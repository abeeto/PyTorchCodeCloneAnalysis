import Adafruit_BBIO.GPIO as GPIO
import time
import os

GPIO.setup("P9_12", GPIO.IN)
GPIO.setup("P9_11", GPIO.OUT)
state = 0
output = 0
while True:
    state = GPIO.input("P9_12")
    if state == 1:
        GPIO.output("P9_11", GPIO.HIGH)
	#os.system('./autocap.sh')        
	#print "button pressed!"
	#print "Ready!"
    if state == 0:
        GPIO.output("P9_11", GPIO.LOW)
    time.sleep(0.1)
