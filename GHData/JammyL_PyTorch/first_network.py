import torch
import torchvision
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np 
import tqdm as tqdm
import numpy.random as r
import time as time
from torchvision import transforms, datasets


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


class Net(nn.Module):
	def __init__(self, input_size = 28*28, output_size = 10):
		super().__init__()
		

		self.layer_1 = nn.Linear(input_size, 32)
		self.layer_2 = nn.Linear(32, 64)
		self.layer_3 = nn.Linear(64, 128)
		self.layer_4 = nn.Linear(128, output_size)

		self.optimizer = optim.Adam(self.parameters(), lr=0.001)
		self.loss_function = nn.MSELoss()

	def fwd_prop(self, data):
		data = F.relu(self.layer_1(data))
		data = F.relu(self.layer_2(data))
		data = F.relu(self.layer_3(data))
		data = self.layer_4(data)
		return F.log_softmax(data, dim=-1)

	def train(self, trainset, epochs):
		for epoch in range(epochs):
			self.optimizer = optim.Adam(self.parameters(), lr=(epochs + 1 - epoch)*0.001)
			for data in trainset:
				image, value = data
				image = image.to(device)
				self.zero_grad()
				oh_value = torch.nn.functional.one_hot(value, 10).to(device).type(dtype = torch.float)
				output = self.fwd_prop(image.view(-1,784))
				loss = self.loss_function(output, F.log_softmax(oh_value + 0.001, dim = -1))
				print(loss)
				loss.backward()  
				self.optimizer.step()
				
				print(f"Epoch: {epoch}. Loss: {loss}")

	def test(self, testset):
		correct = 0
		total = 0
		with torch.no_grad():
			for data in testset:
				image, value = data
				image = image.to(device)
				output = self.fwd_prop(image.view(-1,784))

		for idx, i in enumerate(output):
			if torch.argmax(i) == value[idx]:
				correct += 1
			total += 1

		print("Accuracy: ", round(correct/total, 3))

	def test_single(self, testpoint):

		with torch.no_grad():
			n = r.randint(len(testpoint[0]))
			image, value = testpoint
			plt.imshow(image[n][0], cmap = 'Greys')
			image = image[n].to(device)
			output = self.fwd_prop(image.view(-1,784))

			for i in range(len(output[0])):
				print(np.exp(output.cpu()[0][i]))
			
			print('Predicted:', int(torch.argmax(output)))
			print('Actual:', int(value[n]))

			plt.show()















