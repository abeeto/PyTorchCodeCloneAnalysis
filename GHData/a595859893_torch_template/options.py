import torch
import argparse

class Options:
	def __init__(self):
		parser = argparse.ArgumentParser()
		
		parser.add_argument("--device", type=str,
			default = 'cuda:0' if torch.cuda.is_available else 'cpu')
		parser.add_argument("--data_path", type=str,
			default = './data')
		parser.add_argument("--batch", type=int,
			default = 512)
		parser.add_argument("--epoch", type=int,
			default = 4)

		parser.add_argument("--image_shape", type=int, nargs=3,
			default = [1,28,28])

		parser.add_argument("--model_path", type=str,
			default = './save')
		parser.add_argument("--model_load", type=str,
			help = 'empty:new model, latest:latest model, <model_name>:load model with name <model_name>',
			default = 'latest')


		self.kernel_size = (5,5)
		self.kernel_stride = 1
		self.output_dim = 10
		self.args = parser.parse_args()