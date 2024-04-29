import torch

if __name__ == '__main__':
	print("CUDA available: %r" % (torch.cuda.is_available()))