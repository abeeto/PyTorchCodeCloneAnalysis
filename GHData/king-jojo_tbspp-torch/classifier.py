import time
import math
import ast
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tree_dataset import TreeDataSet
from Anode2vec import EMBEDDING_DIM

LEARN_RATE = 0.001
EPOCHS = 5
BATCH_SIZE = 1
OUTPUT_SIZE1 = 240
OUTPUT_SIZE2 = 120

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class tbspp(nn.Module):
	def __init__(self, label_size):
		super(tbspp, self).__init__()
		self.l1 = nn.Linear(3 * EMBEDDING_DIM, OUTPUT_SIZE1)
		self.l2 = nn.Linear(OUTPUT_SIZE1, OUTPUT_SIZE2)
		self.fc = nn.Linear(15 * OUTPUT_SIZE2, label_size)
		

	def forward(self, nodes, children):
		x = self.tree_conv(nodes, children)
		x = torch.tanh(self.l1(x))
		x = torch.tanh(self.l2(x))
		x = self.pyramid_pooling(x)
		x = self.fc(x)
		return x


	def tree_conv(self, nodes, children):
		tree = self.tree_tensor(nodes, children)
		# Total coef
		c_t = self.coef_top(children)
		c_r = self.coef_right(children, c_t)
		c_l = self.coef_left(children, c_t, c_r)
		coef = torch.stack([c_t, c_r, c_l], 3)

		batch_size = children.size(0)
		num_nodes = children.size(1)
		max_children = children.size(2)
		x = batch_size * num_nodes
		y = max_children + 1
		res = tree.view(x, y, EMBEDDING_DIM)
		coef = coef.view(x, y, 3)
		res = torch.matmul(res.permute(0,2,1), coef)
		res = res.view(batch_size, num_nodes, 3 * EMBEDDING_DIM)
		return res

	def pyramid_pooling(self, conv):
		batch_size = conv.size(0)
		node_size = conv.size(1)
		conv = conv.permute(0,2,1)
		# pooling with different kernel size
		pooling1 = nn.MaxPool1d(node_size)
		pooling2 = nn.MaxPool1d(int(node_size/2), int(node_size/2))
		pooling3 = nn.MaxPool1d(int(node_size/4), int(node_size/4))
		pooling4 = nn.MaxPool1d(int(node_size/8), int(node_size/8))
		pool1 = pooling1(conv).view((batch_size,-1))
		pool2 = pooling2(conv).view((batch_size,-1))
		pool3 = pooling3(conv).view((batch_size,-1))
		pool4 = pooling4(conv).view((batch_size,-1))
		return torch.cat([pool1, pool2, pool3, pool4], 1)


	def tree_tensor(self, nodes, children):
		batch_size = children.size(0)
		num_nodes = children.size(1)
		max_children = children.size(2)

		# Replace root vector with zero vector
		zero_vecs = torch.zeros((batch_size, 1, EMBEDDING_DIM))
		vector_lookup = torch.cat([zero_vecs, nodes[:, 1:, :]], 1)
		children_tensor = torch.cat(
			              [torch.cat(
			              	[torch.index_select(vector_lookup[0], 0, i).unsqueeze(0) 
			              	for i in batch]).unsqueeze(0) 
			              for batch in children])
		nodes = torch.unsqueeze(nodes, 2)
		tree_tensor = torch.cat([nodes, children_tensor], 2)
		return tree_tensor

	def coef_top(self, children):
		batch_size = children.size(0)
		num_nodes = children.size(1)
		max_children = children.size(2)

		# mask for parent node
		return torch.cat(
			   [torch.ones((num_nodes, 1)), torch.zeros((num_nodes, max_children))], 
			   1).unsqueeze(0).expand(batch_size, num_nodes, max_children+1)

	def coef_right(self, children, c_t):
		children = children.float()
		batch_size = children.size(0)
		num_nodes = children.size(1)
		max_children = children.size(2)

		# number of non-zero numbers, such as [[[6],[7],[13],[5]...[0]]]
		# [[[6,6,6...6],[7,7,7...7],[13,13,13...13],[5,5,5...5]...[0,0,0...0]]]
		num_siblings = torch.cat(
			              [torch.cat(
			              	[torch.tensor([torch.nonzero(i).size(0)], dtype=torch.float32).unsqueeze(0) 
			              	for i in batch]).unsqueeze(0) 
			              for batch in children]).expand(batch_size, num_nodes, max_children+1)

		# mask for nodes [[[0,1,1...0],[0,1,0...0],[0,1,1...0]...[0,0,0...0]]]
		mask = torch.cat(
			   [torch.zeros((batch_size, num_nodes, 1)), 
			    torch.min(children, torch.ones((batch_size, num_nodes, max_children)))], 
			    2)

		# indices: [[[-1,0,1,2...12],[-1,0,1,2...12],[-1,0,1,2...12]...[-1,0,1,2...12]]]
		child_indices = torch.mul(
			torch.arange(-1, max_children, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(batch_size, num_nodes, max_children+1),
			mask
			)

		singles = torch.cat(
			[torch.zeros((batch_size, num_nodes, 1)),
			 torch.full((batch_size, num_nodes, 1), 0.5),
			 torch.zeros((batch_size, num_nodes, max_children-1))
			], 2)

		# if the child is not single, then replace it with  [0,0,1,2,3,4...12]/(num-1)
		# return coef such as [0,0,1/6,2/6,3/6,4/6,5/6,6/6...0]
		return torch.where(
			num_siblings.eq(1.0),
			singles,
			torch.mul((1.0 - c_t), torch.div(child_indices, num_siblings - 1.0))
			)

	def coef_left(self, children, c_t, c_r):
		children = children.float()
		batch_size = children.size(0)
		num_nodes = children.size(1)
		max_children = children.size(2)

		# such as mask = [[[0,1,1...0],[0,1,0...0],[0,1,1...0]...[0,0,0...0]]]
		mask = torch.cat(
			[torch.zeros((batch_size, num_nodes, 1)),
			 torch.min(children, torch.ones((batch_size, num_nodes, max_children)))
			], 2)

		# return the coef of left: such as [0,1,5/6,4/6,3/6,2/6,1/6,0...0]
		return torch.mul(
			   torch.mul((1.0 - c_t), (1.0 - c_r)), mask
			)

def timeSince(since):
	now = time.time()
	s = now - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

def main():
    input_tree = './data/algorithm_trees.pkl'
    input_embed = './data/vectors.pkl'
    with open(input_tree, 'rb') as fh:
        trees, _, labels = pickle.load(fh)

    with open(input_embed, 'rb') as fh:
        embeddings, embed_lookup = pickle.load(fh)

    embeddings_new = nn.Embedding(embeddings.weight.size(0)+1, embeddings.weight.size(1))
    # add one feature for padding
    with torch.no_grad():
        tensor_for_padding = torch.tensor([[0] * EMBEDDING_DIM], dtype=torch.float32)
        embeddings_pad = torch.cat((embeddings.weight, tensor_for_padding), 0)
        embeddings_new.weight.data.copy_(embeddings_pad)

    label_size = len(labels)
    traindata = TreeDataSet(trees, labels, embed_lookup)

    model = tbspp(label_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARN_RATE)
    
    start = time.time()
    print("Waiting...")
    for epoch in range(1, EPOCHS+1):
        total_loss = 0.0
        dataset_size = 0
        for batch in traindata.data_gen():
            nodes, children, label = batch
            nodes2vec = embeddings_new(torch.tensor(nodes, dtype=torch.long))
            children2tensor = torch.tensor(children, dtype=torch.long)

            nodes2vec.to(DEVICE)
            children2tensor.to(DEVICE)

            optimizer.zero_grad()
            out = model(nodes2vec, children2tensor)
            loss = criterion(out, torch.tensor(label, device=DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            dataset_size += 1
        print('(%s) Epoch: %d/%d Loss: %.3f' % (timeSince(start), epoch, EPOCHS, total_loss / dataset_size))

    torch.save(model, './data/tbspp.pth')
    print('Model saved')

    correct = 0
    total = 0
    with torch.no_grad():
    	for batch in traindata.data_gen():
    		nodes, children, label = batch
    		nodes2vec = embeddings_new(torch.tensor(nodes, dtype=torch.long))
    		children2tensor = torch.tensor(children, dtype=torch.long)
    		label = torch.tensor(label, device=DEVICE)
    		nodes2vec.to(DEVICE)
    		children2tensor.to(DEVICE)
    		out = model(nodes2vec, children2tensor)
    		_, predicted = torch.max(out.data, 1)
    		total += label.size(0)
    		correct += (predicted == label).sum().item()
    print('Accuracy of the tework on the training data set is: %.3f %%' % (100* correct/total))

if __name__ == '__main__':
    main()
