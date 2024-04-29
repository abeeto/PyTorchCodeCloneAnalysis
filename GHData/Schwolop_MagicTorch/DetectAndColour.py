'''
DetectAndColour
===============
Detects non-white-ish objects and projects white light on them, but not anywhere else.
'''

import numpy as np
import cv2
import video
from common import draw_str # From common.py in this directory.

inputRes = (640,480)  	# webcame native res
outputRes = (848,480) 	# pico-projector native res
thresh = 100 			# Threshold for white-detection.
blockSize = 5 			# Block size for adaptive thresholding region.
C = 1					# Badly documented parameter for adaptive thresholding.

class App:
	def __init__(self):
		self.cam = cv2.VideoCapture(0) # TODO: Enumerate all possible, and select sensibly.
		self.input = None
		self.output = None

	def run(self):
		cv2.namedWindow("Output")#, cv2.WINDOW_NORMAL)# | cv2.GUI_NORMAL) # GUI_NORMAL not supported for some reason?
		cv2.namedWindow("Input")#, cv2.WINDOW_NORMAL)
		cv2.resizeWindow("Output", *outputRes )
		cv2.resizeWindow("Input", *inputRes )
		
		while True:
			# Read a camera frame and process it.
			ret, frame = self.cam.read()
			self.input = frame.copy()
			frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# Basic thresholding
			ret, frameGray = cv2.threshold( frameGray, thresh, 255, cv2.THRESH_BINARY_INV )
			#frameGray = cv2.adaptiveThreshold( frameGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, C ) # This is essentially an edge detector.
			
			# Display the frame copy (with annotations)
			self.output = cv2.resize(frameGray,outputRes)
			h, w = self.input.shape[:2]
			H, W = self.output.shape[:2]
			cv2.rectangle( self.output, (0,0), (W, H), (0,0,0), -1 ) # Fill window with black.
			self.output[ (H-h)/2 : H - ((H-h)/2), (W-w)/2 : W - ((W-w)/2) ] = frameGray # Draw frame in centre of self.output.
			draw_str(self.output, (20, 40), 'Output')
			draw_str(self.input, (20, 40), 'Input')
			cv2.imshow('Output', self.output) # Show self.output
			cv2.imshow('Input', self.input) # Show self.input

			ch = 0xFF & cv2.waitKey(1)
			if ch == 27: # If Esc, exit
				break

def main():
	print __doc__
	App().run()
	cv2.destroyAllWindows() 			

if __name__ == '__main__':
	main()
