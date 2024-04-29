import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import PIL
from PIL import Image
from flowlib import read
from skimage import io, transform

device='cuda'

def toString4(num):
	string = str(num)
	while(len(string) < 4):
		string = "0"+string
	return string

def toString7(num):
	string = str(num)
	while(len(string) < 7):
		string = "0"+string
	return string

class MPIDataset(Dataset):

	def __init__(self, path, transform=None):
		"""
		looking at the "clean" subfolder for images, might change to "final" later
		root_dir -> path to the location where the "training" folder is kept inside the MPI folder
		"""
		self.path = path+"training/"
		self.transform = transform
		self.dirlist = os.listdir(self.path+"clean/")
		self.dirlist.sort()
		# print(self.dirlist)
		self.numlist = []
		for folder in self.dirlist:
			self.numlist.append(len(os.listdir(self.path+"clean/"+folder+"/")))
		self.length=sum(self.numlist)-len(self.numlist)

	def __len__(self):

		return self.length

	def __getitem__(self, idx):
		"""
		idx must be between 0 to len-1
		assuming flow[0] contains flow in x direction and flow[1] contains flow in y
		"""
		for i in range(0, len(self.numlist)):
			folder = self.dirlist[i]
			path = self.path+"clean/"+folder+"/"
			occpath = self.path+"occlusions/"+folder+"/"
			flowpath = self.path+"flow/"+folder+"/"
			if(idx < (self.numlist[i]-1)):
				num1 = toString4(idx+1)
				num2 = toString4(idx+2)
				img1 = io.imread(path+"frame_"+num1+".png")
				img2 = io.imread(path+"frame_"+num2+".png")
				mask = io.imread(occpath+"frame_"+num1+".png")
				img1 = torch.from_numpy(transform.resize(img1, (360, 640))).to(device).permute(2, 0, 1).float()
				img2 = torch.from_numpy(transform.resize(img2, (360, 640))).to(device).permute(2, 0, 1).float()
				mask = torch.from_numpy(transform.resize(mask, (360, 640))).to(device).float()
				flow = read(flowpath+"frame_"+num1+".flo")
				# bilinear interpolation is default
				originalflow=torch.from_numpy(flow)                
				flow = torch.from_numpy(transform.resize(flow, (360, 640))).to(device).permute(2,0,1).float()
				flow[0, :, :] *= float(flow.shape[1])/originalflow.shape[1]
				flow[1, :, :] *= float(flow.shape[2])/originalflow.shape[2]
				# print(flow.shape) #y,x,2
				# print(img1.shape)
				break

			idx -= self.numlist[i]-1

		if self.transform:
			# complete later
			pass
#IMG2 should be at t in IMG1 is at T-1
		return (img1, img2, mask, flow)

class FlyingChairsDataset(Dataset):

	def __init__(self, path, transform=None):
		"""
		looking at the "clean" subfolder for images, might change to "final" later
		root_dir -> path to the location where the "training" folder is kept inside the MPI folder
		"""
		self.path = path+"train/"
		self.transform = transform
		self.length=22230 #14 files corresponding to each image pair

	def __len__(self):

		return self.length

	def __getitem__(self, idx):
		"""
		idx must be between 0 to len-1
		assuming flow[0] contains flow in x direction and flow[1] contains flow in y
		"""
		num = toString7(idx)
		img1 = io.imread(self.path+num+"-img_0.png")
		img2 = io.imread(self.path+num+"-img_1.png")
		mask = io.imread(self.path+num+"-mb_01.png")
		img1 = torch.from_numpy(transform.resize(img1, (360, 640))).to(device).permute(2, 0, 1).float()
		img2 = torch.from_numpy(transform.resize(img2, (360, 640))).to(device).permute(2, 0, 1).float()
		mask = torch.from_numpy(transform.resize(mask, (360, 640))).to(device).float()
		flow = read(self.path+num+"-flow_01.flo")
		# bilinear interpolation is default
		originalflow=torch.from_numpy(flow)                
		flow = torch.from_numpy(transform.resize(flow, (360, 640))).to(device).permute(2,0,1).float()
		flow[0, :, :] *= float(flow.shape[1])/originalflow.shape[1]
		flow[1, :, :] *= float(flow.shape[2])/originalflow.shape[2]
		
		if self.transform:
			# complete later
			pass
#IMG2 should be at t in IMG1 is at T-1
		return (img1, img2, mask, flow)

class ConsolidatedDataset(Dataset):

	def __init__(self, MPI_path, FC_path, transform=None):
		self.mpi=MPIDataset(MPI_path)
		self.fc=FlyingChairsDataset(FC_path)
		
	def __len__(self):

		return len(self.mpi)+len(self.fc)

	def __getitem__(self, idx):
		"""
		idx must be between 0 to len-1
		"""
		if(idx<len(self.mpi)):
			return self.mpi.__getitem__(idx)
		else:
			return self.fc.__getitem__(idx-len(self.mpi))

