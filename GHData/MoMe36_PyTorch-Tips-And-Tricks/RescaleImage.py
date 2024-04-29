import torch 
import numpy as np 
from torchvision import transforms 
from skimage import transform 


class Rescale: 

	def __init__(self, output_size): 
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size
	
	def __call__(self, tensor): 


		h,w = tensor.shape[-2:]
		new_h, new_w = self.output_size

		img = transform.resize(tensor, (new_h, new_w))
		return img 

class ToTensor: 

	def __call__(self, sample): 
		
		image = sample.transpose((2,0,1))
		return torch.tensor(image).float()


scale = Rescale((224,224))
tensorize = ToTensor() 

composed = transforms.Compose([scale, tensorize]) 



x = np.random.uniform(0, 1., (64,64,3))
images = [np.random.uniform(0,1, (64+i, 70+i, 3)) for i in range(20)]
for i in images: 
	print('Initial shape: {}, Transformed shape: {}'.format(i.shape, composed(i).shape))
