
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class Tester:
	def __init__(self,net,loader,verbose=0):
		self.net = net
		self.loader = loader
		self.verbose = verbose

	def run(self,epochs=1):
		total = 0
		correct = 0
		for i in range(0,epochs):
			for data in self.loader:
				images, labels = data
				outputs = self.net(Variable(images))
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum()
				# print("Test images:",images)
				# print("Test predicted:",predicted)
				# print("Test lables:",labels)
				if self.verbose > 0:
					if ((predicted == labels).sum()):
						if predicted[0] == 0:
							print("O",end='')
						else:
							print("o",end='')
					else:
						if predicted[0] == 0:
							print("X",end='')
						else:
							print("x",end='')

		# print('Accuracy of the network on the %d test images: %d %%' % (total, 100 * correct / total))
		return total, correct
