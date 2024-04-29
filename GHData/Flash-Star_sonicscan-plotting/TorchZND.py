import numpy as np
from collections import OrderedDict
import glob

class TorchZND:

	def __init__(self):
		# Torch outputs for ZND calculations take the form X_YYYY_z1.dat, 
		# where X is the prefix and YYYY is a 4-digit set of numbers.
		self.prefix = 'znd'
		
		# Initialize data structures
		self.znd = OrderedDict([])
		# self.vars keys are variable names, values are 2-element lists consisting of [YYYY (string),column # (int)]
		self.vars = OrderedDict([])
		# self.files keys are YYYY (string), values are file objects
		self.files = OrderedDict([])

	def setPrefix(self,p):
		self.prefix = str(p)

	def getPrefix(self):
		return self.prefix

	def readZND(self):
		file_list = glob.glob(self.prefix + '*.dat')
		for f in file_list:
			fl = f.split('_')
			fy = fl[-2]	
			self.files[fy] = open(f,'r')
		for k in self.files.keys():
			# fill self.vars data structure
			self.files[k].readline() # throwaway line
			l = self.files[k].readline() # initial values for detonation, keep them
			l = l.split()
			for i in [3,5,7,9]:
				self.znd[l[i].rstrip('=')] = [float(l[i+1])]
			l = self.files[k].readline() # column names
			l = l.split()
			# create temporary column->var mapping for this file
			c2v = OrderedDict([])
			for i in range(1,len(l)): # ignore leading *
				self.vars[l[i]] = [k,i]
				self.znd[l[i]] = []
				c2v[i] = l[i]
			# as of here, the next readline() will get the first line of data
			# the znd files I'm reading don't suffer from the first data line having different contents at the same time 0
			for l in self.files[k]:
				l = l.split()
				for i in c2v.keys(): # ignore leading *
					self.znd[c2v[i]].append(float(l[i]))
			# close file
			self.files[k].close()
		for k in self.znd.keys():
			# convert lists to numpy arrays
			self.znd[k] = np.array(self.znd[k])

	def getZND(self):
		return self.znd
		
				
	
				

