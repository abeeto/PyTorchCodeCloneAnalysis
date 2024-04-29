#!/usr/bin/env python
# -*- coding: utf-8 -*-

# IMU Utility Functions

def quaternionToEuler(x,y,z,w):
	from math import atan2, asin
	roll  = atan2(2*y*w - 2*x*z, 1 - 2*y*y - 2*z*z)
	pitch = atan2(2*x*w - 2*y*z, 1 - 2*x*x - 2*z*z)
	yaw   =  asin(2*x*y + 2*z*w)
	return roll,pitch,yaw

def quaternionToOpenGLMatrix(x,y,z,w):
	matrix = [[1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y), 0],
	          [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x), 0],
	          [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y), 0],
	          [                0,                 0,                 0, 1]]
	return matrix