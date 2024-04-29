import torch
from torch.utils.data import Dataset
from node_map import NODE_MAP

class NodeDataSet(Dataset):
	""" Build DataSet for vectorize """
	def __init__(self, x):
		self.children, self.parent = self.sampler(x)
		self.len = len(self.children)
		self.train = torch.tensor(self.children, dtype=torch.long)
		self.label = torch.tensor(self.parent, dtype=torch.long)

	def __getitem__(self, index):
		return self.train[index], self.label[index]

	def __len__(self):
		return self.len

	def sampler(self, samples):
		index_of = lambda x: NODE_MAP[x]
		children, parent = [], []
		for sample in samples:
			if sample['parent'] is not None:
				children.append(index_of(sample['node']))
				parent.append(index_of(sample['parent']))
		return children, parent