
import torch
import torch.nn as nn
import torch.nn.functional as F

#
# Simple (2 convolution layers) CNN
#
class CNN(nn.Module):
	def __init__(self,channels=3,px=64,outputs=2):
		super(CNN, self).__init__()
		self.px = px
		self.channels = channels
		self.conv_channels = 6
		self.conv_channels2 = 16
		self.kernel_size = 5
		self.poolingsize = 2
		self.outputs = outputs
		self.linear_px = int((((self.px + 1 - self.kernel_size)/self.poolingsize) + 1 -self.kernel_size)/self.poolingsize)
		self.conv1 = nn.Conv2d(self.channels, self.conv_channels, self.kernel_size)
		self.pool = nn.MaxPool2d(self.poolingsize, self.poolingsize)
		self.conv2 = nn.Conv2d(self.conv_channels, self.conv_channels2, self.kernel_size)
		self.fc1 = nn.Linear(self.conv_channels2 * self.linear_px * self.linear_px, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, self.outputs)

	def forward(self, x):
		# print("input:",x);
		# print("forw 1:",x.size());
		x = self.pool(F.relu(self.conv1(x)))
		# print("forw 2:",x.size(),"e:",int((self.px + 1 - self.kernel_size)/self.poolingsize));
		x = self.pool(F.relu(self.conv2(x)))
		# print("forw 3:",x.size(),"e:",(((self.px + 1 - self.kernel_size)/self.poolingsize) + 1 -self.kernel_size)/self.poolingsize );
		# print("linear_px",self.linear_px)
		# x = x.view(-1, 16 * 5 * 5)
		x = x.view(-1, self.conv_channels2 * self.linear_px * self.linear_px)
		# print("forw 4:",x.size());
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		# print("forward:",x);
		return x

#
# Neural Network of convolutioan layers followed by linear layers
# square image input assumed
#
class NNorCNN(nn.Module):
	def __init__(self,channels=3,px=64,clayers=[(5,6,2),(5,16,2)],layers=[120,84,2],convolution_dropout = 0.0,linear_dropout = 0.0):
		super(NNorCNN, self).__init__()
		self.px = px
		self.channels = channels
		self.layers = layers
		self.clayers = clayers
		lastsize = self.channels * self.px * self.px
		self.convolutions = nn.ModuleList()
		self.convolution_dropout = convolution_dropout
		self.linear_dropout = linear_dropout
		if self.convolution_dropout > 0.0:
			self.convolutions_drop = nn.ModuleList()
		self.pools = nn.ModuleList()
		self.linears = nn.ModuleList()
		if self.linear_dropout > 0.0:
			self.linears_drop = nn.ModuleList()
		nchan = self.channels
		lastpx = self.px
		for l in self.clayers:
			self.convolutions.append(nn.Conv2d(nchan, l[1], l[0]))
			if self.convolution_dropout > 0.0:
				self.convolutions_drop.append(nn.Dropout2d(p=self.convolution_dropout))
			self.pools.append(nn.MaxPool2d(l[2],l[2]))
			nchan = l[1]
			lastpx = int((lastpx+1-l[0])/l[2])
		lastsize = nchan * lastpx * lastpx
		self.first_linear_size = lastsize
		for l in self.layers:
			self.linears.append(nn.Linear(lastsize, l))
			if self.linear_dropout > 0.0:
				self.linears_drop.append(nn.Dropout(p=self.linear_dropout))
			lastsize = l

	def forward(self, x):
		for i, l in enumerate(self.convolutions):
			x = F.relu(self.convolutions[i](x))
			if self.convolution_dropout > 0.0:
				x = self.convolutions_drop[i](x)
			x = self.pools[i](x)
		x = x.view(-1,self.first_linear_size)
		last = len(self.layers) - 1
		for i, l in enumerate(self.linears):
			x = l(x)
			if i < last:
				x = F.relu(x)
				if self.linear_dropout > 0.0:
					x = self.linears_drop[i](x)
		return x

#
# Simple fully connected network
#
class NN(nn.Module):
	def __init__(self,channels=3,px=64,layers=[120,84,2]):
		super(NN, self).__init__()
		self.px = px
		self.channels = channels
		self.layers = layers
		lastsize = self.channels * self.px * self.px
		self.linears = nn.ModuleList()
		for l in self.layers:
			self.linears.append(nn.Linear(lastsize, l))
			lastsize = l

	def forward(self, x):
		x = x.view(-1,self.channels * self.px * self.px)
		last = 0
		for i, l in enumerate(self.linears):
			last = i
		for i, l in enumerate(self.linears):
			x = l(x)
			if i < last:
				x = F.relu(x)
		return x
