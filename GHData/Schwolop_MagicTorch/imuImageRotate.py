#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Attempts to stablize an image based on the orientation of an IMU.

''' Calibration Defaults
# Each line starts with "calibration type:"
# followed by the x, y and z calibration, separated by a comma.
# Multiplier and Divider are written as "mul/div"
0: 1000/1019, 1000/1009, 1000/1026
1: 33, 19, 40
2: 500/497, 500/511, 500/441
3: 4, 217, 47
4: 5968/5946, 2768/2686, 2451/2440
5: -9, 20, -5, 1670, -3, -7, 5, 2709
'''

''' Magnetometer calibrated with IMU strapped to projector (off).
# Each line starts with "calibration type:"
# followed by the x, y and z calibration, separated by a comma.
# Multiplier and Divider are written as "mul/div"
0: 1000/1019, 1000/1009, 1000/1026
1: 33, 19, 40
2: 500/316, 500/387, 500/528
3: -316, -388, 347
4: 5968/5946, 2768/2686, 2451/2440
5: -9, 20, -5, 1670, -3, -7, 5, 2709
'''

import cv2, math
from tinkerforge.ip_connection import IPConnection
from tinkerforge.brick_imu import IMU

class TomIMU:	
	def __init__(self,host,port,uid,callbackPeriodMS=100):
		self.host=host
		self.port=port
		self.uid=uid
		self._imu = IMU(uid) # Create device object
		self._ipcon = IPConnection(self.host,self.port)  # Create IPconnection to brickd
		self._ipcon.add_device(self._imu) # Add device to IP connection
		
		self.ready = True # Don't use device before it is added to a connection
		
		# Set period for quaternion callback (defaults to 100ms)
		self._imu.set_quaternion_period(callbackPeriodMS)
		
		# Register quaternion callback
		self._imu.register_callback(self._imu.CALLBACK_QUATERNION, self._QuaternionCallback)
		
		self._imu.leds_off() # Turn LEDs off.
		self._imu.set_convergence_speed(5) # 5ms convergence.
		
		# Orientation origin and most recent values
		q = self._imu.get_quaternion() # Get a temp quaternion from current pose.
		self.rel_x = q.x
		self.rel_y = q.y
		self.rel_z = q.z
		self.rel_w = q.w
		self.x = q.x
		self.y = q.y
		self.z = q.z
		self.w = q.w
		
	def __destroy__(self):
		self._ipcon.destroy()
	
	def _QuaternionCallback(self, x, y, z, w):
		""" Records the most recent quaternion orientation values. """
		self.x,self.y,self.z,self.w = x,y,z,w
		
	def SetOrientationOrigin(self, origin=None):
		""" Resets the orientation origin to the given values, or the latest reading if none. """
		if origin is None:
			self.rel_x, self.rel_y, self.rel_z, self.rel_w = self.x, self.y, self.z, self.w
		else:
			self.rel_x, self.rel_y, self.rel_z, self.rel_w = origin
	
	def GetEulerOrientation(self):
		x,y,z,w = self.GetQuaternionOrientation()
		from math import atan2, asin
		roll  = atan2(2.0*y*w - 2.0*x*z, 1.0 - 2.0*y*y - 2.0*z*z)
		pitch = atan2(2.0*x*w - 2.0*y*z, 1.0 - 2.0*x*x - 2.0*z*z)
		yaw   =  asin(2.0*x*y + 2.0*z*w)
		return roll,pitch,yaw
	
	def GetQuaternionOrientation(self):
		# Conjugate
		x,y,z = -self.x, -self.y, -self.z
		w = self.w
		# Multiply
		wn = w * self.rel_w - x * self.rel_x - y * self.rel_y - z * self.rel_z
		xn = w * self.rel_x + x * self.rel_w + y * self.rel_z - z * self.rel_y
		yn = w * self.rel_y - x * self.rel_z + y * self.rel_w + z * self.rel_x
		zn = w * self.rel_z + x * self.rel_y - y * self.rel_x + z * self.rel_w
		return xn,yn,zn,wn
	
	def GetConvergenceSpeed(self):
		return self._imu.get_convergence_speed()
	
	def SetConvergenceSpeed(self, speed):
		self._imu.set_convergence_speed(speed)

if __name__ == "__main__":
	HOST = "localhost"
	PORT = 4223
	UID = "a4JritAp6Go" # Change to your UID
	outputResolution = (848,480)
	outputFOV = (45,45) # degrees.
	
	imu = TomIMU(HOST,PORT,UID,1)
	
	print "Press Esc to exit..."
	windowName = "IMU Image Stabilization"
	cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)# | cv2.GUI_NORMAL) # GUI_NORMAL not supported for some reason?
	cv2.resizeWindow(windowName, *outputResolution ) # pico-projector native res.
	cam = cv2.VideoCapture(0)
	while(1):
		ret, frame = cam.read()
		output = frame.copy()
		output = cv2.resize(output,outputResolution)
		r,p,y = imu.GetEulerOrientation()
		h, w = frame.shape[:2] # Get height and width of camera frame.
		H,W = output.shape[:2] # Get height and width of projector view.
		
		# Rotate image to match IMU:
		#rotationMatrix = cv2.getRotationMatrix2D( (w/2,h/2), r*180.0/math.pi, 1.0 )
		
		# Counter-rotate the image (assuming IMU + display 
		# are ridigly connected, but camera IS NOT):
		rotationMatrix = cv2.getRotationMatrix2D( (w/2,h/2), -r*180.0/math.pi, 1.0 )
		
		# Counter-rotate the image (assuming IMU + camera + 
		# display are rigidly connected) to keep it upright.
		#rotationMatrix = cv2.getRotationMatrix2D( (w/2,h/2), -r*2.0*180.0/math.pi, 1.0 )
		
		# Shift the image up/down proportional to the pitch.
		udShift = math.floor( (-p*180.0/math.pi / outputFOV[1]) * H )
		# Shift the image left/right proportional to the yaw.
		lrShift = math.floor( (-y*180.0/math.pi / outputFOV[0]) * W )
		#print "U/D: " + str(udShift) + ", L/R: " + str(lrShift)
		
		rotatedFrame = cv2.warpAffine(frame, rotationMatrix, (w,h))
		#print "Roll: " + str(r*180.0/math.pi) + ", Pitch: " + str(p*180.0/math.pi) + ", Yaw: " + str(y*180.0/math.pi)
		#cv2.imshow(windowName, rotatedFrame)
		
		cv2.rectangle( output, (0,0), (W,H), (0,0,0), -1 ) # Fill window with black.
		#output[ (H-h)/2 : H - ((H-h)/2), (W-w)/2 : W - ((W-w)/2) ] = rotatedFrame # Draw frame in centre of output.
		tY = min( H, max( 0, (H-h)/2 - udShift ) )
		bY = max( 0, min( H, H - ((H-h)/2) - udShift ) )
		hY = bY-tY
		lX = min( W, max( 0, (W-w)/2 - lrShift ) )
		rX = max( 0, min( W, W - ((W-w)/2) - lrShift ) )
		wX = rX - lX
		if hY > 0 and wX > 0:
			if udShift > 0 and lrShift > 0:
				output[ tY : bY, lX : rX ] = rotatedFrame[ h-hY : h, w-wX : w ]
			elif udShift > 0 and lrShift <= 0:
				output[ tY : bY, lX : rX ] = rotatedFrame[ h-hY : h, 0 : wX ]
			elif udShift <= 0 and lrShift > 0:
				output[ tY : bY, lX : rX ] = rotatedFrame[ 0 : hY, w-wX : w ]
			elif udShift <= 0 and lrShift <= 0:
				output[ tY : bY, lX : rX ] = rotatedFrame[ 0 : hY, 0 : wX ]
			else:
				print "humpf"
		cv2.imshow(windowName, output)
		
		key = 0xFF & cv2.waitKey(1)
		if key == 27: #Esc to quit.
			break
		elif key == ord(' '): # If spacebar, reset orientation origin.
			imu.SetOrientationOrigin()
		elif key == ord('+'):
			imu.SetConvergenceSpeed( min( 1000, imu.GetConvergenceSpeed() + 5 ) )
			print "IMU convergence speed increased to " + str(imu.GetConvergenceSpeed()) + "ms."
		elif key == ord('-'):
			imu.SetConvergenceSpeed( max( 1, imu.GetConvergenceSpeed() - 5 ) )
			print "IMU convergence speed decreased to " + str(imu.GetConvergenceSpeed()) + "ms."
	cv2.destroyAllWindows()

