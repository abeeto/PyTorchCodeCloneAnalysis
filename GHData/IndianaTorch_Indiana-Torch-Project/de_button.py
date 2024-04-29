import Adafruit_BBIO.GPIO as GPIO
import time
import os

# GPIO pins
GPIO.setup("P9_12", GPIO.IN)
GPIO.setup("P9_11", GPIO.OUT)

# Initialise variables
ledState = 1
buttonState = 0
lastButtonState = 0
reading  = 0

lastDebounceTime = 0
debounceDelay = 0.05

# main loop

while True:

    reading = GPIO.input("P9_12")
    
    if reading != lastButtonState:
        lastDebounceTime = time.time();
        
        if buttonState == 1:
            ledState = 1 - ledState
            print "Pressed!"
  
    if (time.time() - lastDebounceTime) > debounceDelay:
        buttonState = reading
  
    if ledState == 1:
        GPIO.output("P9_11", GPIO.LOW)
        
    if ledState == 0:
        GPIO.output("P9_11", GPIO.HIGH)
        

    lastButtonState = reading
