import RPi.GPIO as GPIO
import subprocess, signal
import os

GPIO.setmode(GPIO.BCM)
GPIO.setup(4,GPIO.IN)

subprocess.call('python clay_pi.py', shell=True)
subprocess.Popen('processing-java --sketch=clay --run', shell=True)

while True:
        try:
                if GPIO.input(4)== 1:
                        subprocess.call('python clay_pi.py', shell=True)
        except ValueError:
                print "Nothing to worry"
