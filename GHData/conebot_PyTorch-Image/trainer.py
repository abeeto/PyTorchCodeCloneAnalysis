
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision

class Trainer:
	def __init__(self,net,loader,optimizer=None,criterion=nn.CrossEntropyLoss(),saveimages=False,verbose=0):
		self.net = net
		self.loader = loader
		if optimizer:
			self.optimizer = optimizer
		else:
			self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
		self.criterion = criterion
		self.saveimages = saveimages
		self.verbose = verbose

	def run(self,epochs=50):
		tot = 0
		for epoch in range(epochs):
			epoch_tot = 0
			epoch_loss = 0
			for i, data in enumerate(self.loader,0):
				inputs, labels = data

				if self.saveimages:
					torchvision.utils.save_image(inputs,"temp-images/train-%d-%d.jpg" % (epoch, i))

				origlabels = labels
				inputs, labels = Variable(inputs), Variable(labels)
				self.optimizer.zero_grad()
				outputs = self.net(inputs)
				# print("Train outputs:",outputs)
				_, predicted = torch.max(outputs.data, 1)

				# print("Train predicted:",predicted)
				# print("Train lables:",labels)

				if self.verbose > 0:
					if ((predicted == origlabels).sum()):
						print("O",end='')
					else:
						print("X",end='')
				loss = self.criterion(outputs, labels)
				epoch_loss += loss.data[0]
				loss.backward()
				self.optimizer.step()

				tot = tot + 1
				epoch_tot += 1
			if self.verbose > 0:
				print(" ",end='')
				print('[%d, %5d] loss: %.4f' % (epoch + 1, i + 1, epoch_loss / epoch_tot))
		return (tot, epoch_loss / epoch_tot)
