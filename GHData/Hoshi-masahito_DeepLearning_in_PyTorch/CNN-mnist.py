import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Hyper Paramater
num_epochs = 50
num_classes = 10
batch_size = 50
learning_rate = 0.001

#Dataset
train_dataset = MNIST('./', train=True, download=True, transform=transforms.ToTensor())

test_dataset = MNIST('./', train=False, download=True, transform=transforms.ToTensor())

#Data Loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

random.seed(0)
numpy.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

class ConvNet(nn.Module):
	def __init__ (self, num_classes=10):
		super(ConvNet, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),#28*28*16
			nn.BatchNorm2d(16),
			nn.ReLU())
		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),#28*28*16
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)) #14*14*16
		self.layer3 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),#14*14*32
			nn.BatchNorm2d(32),
			nn.ReLU())
		self.layer4 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),#16*16*32
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))#7*7*32
		self.layer5 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),#8*8*64
			nn.BatchNorm2d(64),
			nn.ReLU())
		self.layer6 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),#8*8*64
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))#4*4*64
		self.fc1 = nn.Sequential(
			nn.Linear(4*4*64, 2048),
			nn.ReLU(),
			nn.Dropout2d(p=0.5))
		self.fc2 = nn.Sequential(
			nn.Linear(2048, num_classes),
			nn.Dropout2d(p=0.5))

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.layer5(out)
		out = self.layer6(out)
		out = out.reshape(out.size(0), -1)
		out = self.fc1(out)
		out = self.fc2(out)
		return out

model = ConvNet(num_classes).to(device)

#Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#Training
total_step = len(train_loader)
for epoch in range(num_epochs):
	for i , (images, labels) in enumerate(train_loader):
		images = images.to(device)
		labels = labels.to(device)
		
		#Forward pass
		outputs = model(images)
		loss = criterion(outputs, labels)
		
		#Backward and optimize
		optimizer.zero_grad()
		loss.backword()
		optimizer.step()
		
		if (i+1) % 100 == 0:
			print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

#Test the model
model.eval()
with torch.no_grad():
	correct = 0
	total = 0
	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
	
	print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')