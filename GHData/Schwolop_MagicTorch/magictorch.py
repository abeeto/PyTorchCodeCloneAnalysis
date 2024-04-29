'''
Magic Torch Prototype
===============================
Based on the Lucas-Kanade homography tracker from OpenCV samples.
Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames. Finds homography between reference and current views.

Keys
----
ESC   - exit
SPACE - Lock/unlock projection
r     - toggle RANSAC
'''

import numpy as np
import cv2
import video
from common import draw_str # From common.py in this directory.

lk_params = dict( winSize  = (19, 19), # Size of search winow at each pyramid level
maxLevel = 2,
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) # When the alg terminates

feature_params = dict( maxCorners = 1000, # Passed to GoodFeaturesToTrack
qualityLevel = 0.01,
minDistance = 8,
blockSize = 19 )

minFeaturesBeforeNewKeyframeRequired = 20 # Automatically grab a new keyframe if less than this many features are viable.

'''Traces optical flow between two images, then reverses this to assess which features are back-projectable within some tolerance.'''
def checkedTrace(img0, img1, p0, back_threshold = 1.0):
	# Calculate flow points (p1) from img0 to img1, using p0 as the baseline
	p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

	# Calculate flow points (p0r) from img1 to img0, using p1 as the baseline
	p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

	# Calculate the error in back projection for each point
	d = abs(p0-p0r).reshape(-1, 2).max(-1)
	status = d < back_threshold # If error < threshold, status is True, else False. This builds an array of statuses.
	return p1, status # Return the new positions of the features, and the status array

green = (0, 255, 0)
red = (0, 0, 255)

class App:
	def __init__(self):
		self.cam = cv2.VideoCapture(0) # TODO: Enumerate all possible, and select sensibly.
		self.p0 = None
		self.use_ransac = True
		self.frame0 = None
		self.output = None

	def run(self):
		cv2.namedWindow("MagicTorch", cv2.WINDOW_NORMAL)# | cv2.GUI_NORMAL) # GUI_NORMAL not supported for some reason?
		cv2.resizeWindow("MagicTorch", 848, 480 ) # pico-projector native res.
		while True:
			# Read a camera frame, convert to grayscale, and copy.
			ret, frame = self.cam.read()
			frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			vis = frame.copy()
			if self.output is None:
				self.output = cv2.resize(vis,(848,480))

			# If we have a prior set of features to track...
			if self.p0 is not None:
				# Get the positions (p2) of the features (p1) in the new frame, and their statuses.
				p2, trace_status = checkedTrace(self.gray1, frame_gray, self.p1)

				# Update variables for next iteration, saving only the good features.
				self.p1 = p2[trace_status].copy()
				self.p0 = self.p0[trace_status].copy()
				self.gray1 = frame_gray

				# If there are less than minFeaturesBeforeNewKeyframeRequired features, tracking is impossible, so clear the baseline features (p0)
				if len(self.p0) < minFeaturesBeforeNewKeyframeRequired:
					self.frame0 = frame.copy() # Save the original frame
					self.p0 = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
					if self.p0 is not None:
						self.p1 = self.p0
						self.gray0 = frame_gray
						self.gray1 = frame_gray
						continue

				# Find the perspective transformation between the sets of features.
				H, status = cv2.findHomography(self.p0, self.p1, (0, cv2.RANSAC)[self.use_ransac], 10.0)
				h, w = frame.shape[:2] # Get height and width of camera frame.
				overlay = cv2.warpPerspective(self.frame0, H, (w, h)) # Warp the original frame by this homography matrix
				vis = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0.0) # and draw translucently over the current frame (this has the effect that the original image looks bright, within the dull-by-comparison live feed).

				for (x0, y0), (x1, y1), good in zip(self.p0[:,0], self.p1[:,0], status[:,0]):
					if good:
						cv2.line(vis, (x0, y0), (x1, y1), (0, 128, 0)) # Draw a green line from the original feature to its new location.
						cv2.circle(vis, (x1, y1), 2, (red, green)[good], -1) # Draw a green circle on the new location of good features, and red for bad.
						draw_str(vis, (20, 20), 'track count: %d' % len(self.p1)) # Draw an overlay detailing the amount of tracks.
						if self.use_ransac:
							draw_str(vis, (20, 40), 'RANSAC')

			# Else, if no prior set of features to track...
			else:
				# Find some features in the grayscale image.
				p = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
				if p is not None:
					# Draw them to the frame copy.
					for x, y in p[:,0]:
						cv2.circle(vis, (x, y), 2, green, -1)
						draw_str(vis, (20, 20), 'feature count: %d' % len(p))

			# Display the frame copy (with annotations)
			h,w = vis.shape[:2]
			H,W = self.output.shape[:2]
			cv2.rectangle( self.output, (0,0), (W,H), (0,0,0), -1 ) # Fill window with black.
			self.output[ (H-h)/2 : H - ((H-h)/2), (W-w)/2 : W - ((W-w)/2) ] = vis # Draw vis in centre of self.output.
			cv2.imshow('MagicTorch', self.output) # Show self.output

			ch = 0xFF & cv2.waitKey(1)
			if ch == 27: # If Esc, exit
				break
			if ch == ord(' ') or self.frame0 is None: # If space, use the current frame's features as the baseline
				self.frame0 = frame.copy() # Save the original frame
				self.p0 = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
				if self.p0 is not None:
					self.p1 = self.p0
					self.gray0 = frame_gray
					self.gray1 = frame_gray
			if ch == ord('r'): # If r, toggle whether to use ransac.
				self.use_ransac = not self.use_ransac

def main():
	print __doc__
	App().run()
	cv2.destroyAllWindows() 			

if __name__ == '__main__':
	main()
