import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable

from config import cfg

def add_conv_block(in_ch=1, out_ch=1, filter_size=3, dilate=1, last=False):
	conv_1 = nn.Conv2d(in_ch, out_ch, filter_size, padding=dilate*(1-last), dilation=dilate)
	bn_1 = nn.BatchNorm2d(out_ch)

	return [conv_1, bn_1]

class MSDNet(nn.Module):
	"""
	Paper: A mixed-scale dense convolutional neural network for image analysis
	Published: PNAS, Jan. 2018 
	Paper: http://www.pnas.org/content/early/2017/12/21/1715832114
	"""
	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Linear):
			torch.nn.init.kaiming_normal(m, m.weight.data)

	def __init__(self, num_layers=40, in_channels=None, out_channels=None):
		if in_channels is None:
			in_channels=cfg.IN_CHANNELS

		if out_channels is None:
			out_channels=cfg.N_CLASSES

		super(MSDNet, self).__init__()

		self.layer_list = add_conv_block(in_ch=in_channels)
		
		current_in_channels = 1
		# Add N layers
		for i in range(num_layers):
			s = (i)%10 + 1
			self.layer_list += add_conv_block(in_ch=current_in_channels, dilate=s)
			current_in_channels += 1

		# Add final output block
		self.layer_list += add_conv_block(in_ch=current_in_channels + in_channels, out_ch=out_channels, filter_size=1, last=True)

		# Add to Module List
		self.layers = nn.ModuleList(self.layer_list)

		self.apply(self.weight_init)

	def forward(self, x):
		prev_features = []
		inp = x
		
		for i, f in enumerate(self.layers):
			# Check if last conv block
			if i==(len(self.layers) - 2):
				x = torch.cat(prev_features + [inp], 1)
			
			x = f(x)
			
			if (i+1)%2 == 0 and (not i==(len(self.layers)-1)):
				x = F.relu(x)
				# Append output into previous features
				prev_features.append(x)
				x = torch.cat(prev_features, 1)

		x = F.log_softmax(x)
		return x