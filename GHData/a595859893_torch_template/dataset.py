from torch.utils.data import Dataset
from torchvision import datasets, transforms
from options import Options

class MyDataset(Dataset):
	def __init__(self, options:Options, train=True):
		transform = transforms.Compose([transforms.ToTensor()])
		self.data = datasets.MNIST(
			root = options.args.data_path,
			transform = transform,
			train = train,
			download = True)

	def __len__(self):
		return len(self.data)

	def __getitem__(self,index):
		return self.data[index]
