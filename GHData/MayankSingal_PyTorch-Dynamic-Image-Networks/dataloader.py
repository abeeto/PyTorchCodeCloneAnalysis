import torch
import torch.utils.data as data
import numpy as np
import cv2
from PIL import Image
import glob
import os
import random
from random import shuffle

def create_train_list():

	folderList = sorted(glob.glob("/home/user/data/jpegs_256/*"))
	trainList = []

	classes = []
	for folder in folderList:
		act = folder.split("/")[-1].split("_")[1]
		classes.append(act)
	classes = list(set(classes))

	for i,folder in enumerate(folderList):
		image_list = sorted(glob.glob(folder + "/*.jpg"))
		cat = folder.split("/")[-1].split("_")[1]
		DIpaths = getImages(image_list)

		for DIpath in DIpaths:
			trainList.append([DIpath, classes.index(cat)])

		print(i+1, "folders done!")
		
	return trainList


def populate_train_list():

	with open('trainData.txt') as f:
		lines = f.readlines()
		lines = [line.split("\n")[0] for line in lines]

	train_list = []

	for line in lines:
		tmp = []
		tmp.append(line.split(" ")[0])
		tmp.append(int(line.split(" ")[1]))
		train_list.append(tmp)

	return train_list








def harmonicT(t):
	out = 0.0
	for i in range(1,t+1):
		out += 1/float(i)

	return out

def alphaT(t,T):

	return 2*(T-t+1) - (T+1)*(harmonicT(T) - harmonicT(t-1))


def alphaT2(t,T):

	return 2*t - T - 1


def get_dynamic_image(image_List):

	length = len(image_List)#[:25]

	DI = sum([alphaT(i, length)*np.asarray(Image.open(image_List[i])) for i in range(length)])
	minval = np.min(DI)
	maxval = np.max(DI)
	normalized_DI = (255)*(DI - minval)/(maxval - minval)


	return Image.fromarray(np.uint8(normalized_DI))


def getImages(im_list, window=25, stride=20):

	idx = 0
	out_list = []

	while(idx + window < len(im_list)):
		tmpList = im_list[idx:idx+window]
	
		out_list.append(tmpList)
	
		idx += stride


	return out_list


def createDataset():

	final_list = []

	trainList = create_train_list()
	for i,paths in enumerate(trainList):
		tmpDI = get_dynamic_image(paths[0])
		tmpLabel = paths[1]
		tmpDI.save('/home/user/data/dynamic_images/' + str(i) + '.jpg')
		final_list.append(['/home/user/data/dynamic_images/' + str(i) + '.jpg', tmpLabel])

		if (i+1)%100 == 0:
			print(i, "/", len(trainList), "images done!")
			

	return final_list





class dynamic_image_loader(data.Dataset):

	def __init__(self, transform=None, mode='train'):
		self.trainList = populate_train_list() #create_train_list()
		self.transform = transform

		shuffle(self.trainList)

		if mode=='train':
			self.trainList = self.trainList[:-20000]
		else:
			self.trainList = self.trainList[-20000:]
		

	def __getitem__(self, index):
		currDIPath = self.trainList[index][0]
		currLabel = np.array(self.trainList[index][1])

		#currDI = get_dynamic_image(currDIPaths)
		currDI = Image.open(currDIPath)

		if self.transform is not None:
			currDI = self.transform(currDI)


		return currDI, torch.from_numpy(currLabel)

	def __len__(self):
		return len(self.trainList)









#final_list = createDataset()
#np.savetxt("trainData.txt", final_list, fmt="%s")











'''
#trainList = create_train_list()	
#print(len(trainList))
#print(trainList[0])

img = '/home/user/data/jpegs_256/v_ApplyEyeMakeup_g01_c01/frame000001.jpg'
tmp1 = cv2.imread(img)
tmp2 = np.asarray(Image.open(img))

print(tmp1)
print("####")
print(tmp2)

print(np.mean(tmp1))
print(np.mean(tmp2))
'''