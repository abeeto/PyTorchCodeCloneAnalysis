import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F	
import torch.optim as optim
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 50) # 2 Input noses, 50 in middle layers
        self.fc2 = nn.Linear(50, 1) # 50 middle layer, 1 output nodes
	self.rl1 = nn.ReLU()
	self.rl2 = nn.ReLU()
	
    def forward(self, x):
        x = self.fc1(x)
	x = self.rl1(x)
	x = self.fc2(x)
	x = self.rl2(x)
        return x

if __name__ == "__main__":
	## Create Network

	net = Net()
	#print net

	## Optimization and Loss

	#criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
	criterion = nn.MSELoss()
	#criterion = nn.L1Loss()
	#criterion = nn.NLLLoss()
	#criterion = nn.BCELoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.1)
	#optimizer = optim.Adam(net.parameters(), lr=0.01)

	trainingdataX = [[[0.01, 0.01], [0.01, 0.90], [0.90, 0.01], [0.95, 0.95]], [[0.02, 0.03], [0.04, 0.95], [0.97, 0.02], [0.96, 0.95]]]
	trainingdataY = [[[0.01], [0.90], [0.90], [0.01]], [[0.04], [0.97], [0.98], [0.1]]]
 	NumEpoches = 20000
	for epoch in range(NumEpoches):

		running_loss = 0.0
		for i, data in enumerate(trainingdataX, 0):
			inputs = data
			labels = trainingdataY[i]
			inputs = Variable(torch.FloatTensor(inputs))
			labels = Variable(torch.FloatTensor(labels))
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()        
			optimizer.step()
			running_loss += loss.data[0]
			if i % 1000 == 0:
				print "loss: ", running_loss
				running_loss = 0.0
	print "Finished training..."
	print net(Variable(torch.FloatTensor(trainingdataX[0])))
		
	
	


