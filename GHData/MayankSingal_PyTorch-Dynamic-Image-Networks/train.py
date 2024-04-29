import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from dataloader import dynamic_image_loader



def train_val():

	model_ft = models.resnet18(pretrained=True)
	num_ftrs = model_ft.fc.in_features
	model_ft.fc = nn.Linear(num_ftrs, 101)
	model_ft = model_ft.cuda()


	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

	train_transform = transforms.Compose([
		transforms.Scale(256),
		transforms.RandomCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize])
	val_transform = transforms.Compose([
		transforms.Scale((224,224)),
		#transforms.RandomCrop(224),
		transforms.ToTensor(),
		normalize])

	trainDataFeeder = dynamic_image_loader(transform = train_transform)
	valDataFeeder = dynamic_image_loader(transform = val_transform, mode='test')
	train_loader = torch.utils.data.DataLoader(trainDataFeeder, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(valDataFeeder, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)



	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.001)



	for epoch in range(5):
		model_ft.train()
		for i, (DIs, labels) in enumerate(train_loader):

			DIs = DIs.cuda()
			labels = labels.cuda()

			optimizer.zero_grad()
			
			outputs = model_ft(DIs)
			_, preds = torch.max(outputs, 1)

			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			if i%10 == 0:
				print("Iter:", i, "/", len(train_loader))
				print("Loss:", loss.item(), "Accuracy:", (torch.sum(preds == labels.data).data.cpu().numpy()/64).item())

				
		#model_ft.eval()
		running_loss = 0.0
		running_corrects = 0
		for j, (DIs, labels) in enumerate(val_loader):

			DIs = DIs.cuda()
			labels = labels.cuda()

			outputs = model_ft(DIs)
			_, preds = torch.max(outputs, 1)
			loss = criterion(outputs, labels)

			running_loss += loss.item()
			running_corrects += torch.sum(preds == labels.data)


		epoch_loss_val = running_loss / len(val_loader)
		epoch_accuracy_val = running_corrects.double() / (len(val_loader)*64)

		print("######")
		print("Epoch:", epoch)
		print("Val Loss:", epoch_loss_val)
		print("Val Accuracy:", epoch_accuracy_val.item())
		print("######")














train_val()