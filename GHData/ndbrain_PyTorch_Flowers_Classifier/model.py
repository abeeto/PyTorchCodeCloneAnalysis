from torch import nn
from torchvision import models

class Model(object):
	"""docstring for Model"""
	def __init__(self):
		super(Model, self).__init__()

		model = models.vgg19(pretrained=True)
		model.classifier = nn.Sequential(
																			nn.Linear(25088, 4096),
		                                  nn.ReLU(),
		                                  nn.Dropout(0.5),
		                                  nn.Linear(4096, 4096),
		                                  nn.ReLU(),
		                                  nn.Dropout(0.5),
		                                  nn.Linear(4096, 102))

		# this ensure to train the last conv layer & the classifier layer only
		c = 37
		for p in model.parameters():
		  p.requires_grad=False
		  c-=1
		  if c==15:
		    break

		return model