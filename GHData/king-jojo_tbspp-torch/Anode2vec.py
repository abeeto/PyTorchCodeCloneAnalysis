import time
import math
import ast
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader
from node_map import NODE_MAP, NODE_SIZE
from node_dataset import NodeDataSet

EMBEDDING_DIM = 30
BATCH_SIZE = 256
EPOCHS = 50
LEARN_RATE = 0.01

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Anode2vec(nn.Module):
	""" Mapping AST nodes to vectors """
	def __init__(self):
		super(Anode2vec, self).__init__()
		self.embeddings = nn.Embedding(NODE_SIZE, EMBEDDING_DIM)
		self.linear1 = nn.Linear(EMBEDDING_DIM, 240)
		self.linear2 = nn.Linear(240, 120)
		self.linear3 = nn.Linear(120, NODE_SIZE)

	def forward(self, x):
		embeds = self.embeddings(x) 
		x = torch.tanh(self.linear1(embeds))
		x = torch.tanh(self.linear2(x))
		x = self.linear3(x)
		return self.embeddings, x

def timeSince(since):
	now = time.time()
	s = now - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

def main():
	infile = './data/algorithm_nodes.pkl'
	outfile = './data/vectors.pkl'
	with open(infile, 'rb') as sample_file:
		samples = pickle.load(sample_file)

	trainset = NodeDataSet(samples)
	sample_loader = DataLoader(dataset = trainset,
		                       batch_size = BATCH_SIZE,
		                       shuffle = True,
		                       num_workers = 2)

	loss_function = nn.CrossEntropyLoss()
	model = Anode2vec()
	optimizer = optim.SGD(model.parameters(), lr=LEARN_RATE)

	start = time.time()
	print("Waiting...")
	for epoch in range(1, EPOCHS+1):
		total_loss = 0
		dataset_size = 0
		for batch in sample_loader:
			input_batch, label_batch = batch
			input_batch.to(DEVICE)
			label_batch.to(DEVICE)

			optimizer.zero_grad()
			embed, vec = model(input_batch)
			loss = loss_function(vec, label_batch)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
			dataset_size += 1
		print('(%s) Epoch: %d/%d Loss: %.3f' % (timeSince(start) ,epoch, EPOCHS, total_loss / dataset_size))
	embed_file = open(outfile, 'wb')
	pickle.dump((embed, NODE_MAP), embed_file)
	print('Embedding saved.')
	torch.save(model, './data/Anode2vec.pth')
	print('Model saved.')	

if __name__ == '__main__':
	main()
