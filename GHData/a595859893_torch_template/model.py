import torch.nn as nn
from options import Options

class MyModel(nn.Module):
	def __init__(self, options:Options):
		super().__init__()
		[channel,width,height] = options.args.image_shape
		(kernel_width, kernel_heigth) = options.kernel_size
		kernel_stride = options.kernel_stride
		output_dim = options.output_dim

		latent_x = (width - kernel_width) // kernel_stride + 1
		latent_y = (height - kernel_heigth) // kernel_stride + 1
		latent_dim = latent_x * latent_y
		self.conv = nn.Conv2d(channel, 1, (kernel_width, kernel_heigth), kernel_stride)
		print(latent_dim, output_dim)
		self.linear = nn.Linear(latent_dim, output_dim)

	def forward(self,x):
		output = self.conv(x)
		output = output.view(output.shape[0],-1)
		output = self.linear(output)
		return output