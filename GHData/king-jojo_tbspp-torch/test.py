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
from classifier import tbspp
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
	tbspp = torch.load('./data/tbspp.pth')
	with open('./data/algorithm_trees.pkl', 'rb') as fh:
		_, trees, labels = pickle.load(fh)
	with open('./data/vectors.pkl', 'rb') as fh:
		embeddings, embed_lookup = pickle.load(fh)

	embeddings_new = nn.Embedding(embeddings.weight.size(0)+1, embeddings.weight.size(1))
	# add one feature for padding
	with torch.no_grad():
		tensor_for_padding = torch.tensor([[0] * EMBEDDING_DIM], dtype=torch.float32)
		embeddings_pad = torch.cat((embeddings.weight, tensor_for_padding), 0)
		embeddings_new.weight.data.copy_(embeddings_pad)

	label_size = len(labels)
	testdata = TreeDataSet(trees, labels, embed_lookup)

	# correct = 0
	# total = 0
	predict_list = []
	correct_list = []
	with torch.no_grad():
		for batch in testdata.data_gen():
			nodes, children, label = batch
			nodes2vec = embeddings_new(torch.tensor(nodes, dtype=torch.long))
			nodes2vec.to(DEVICE)
			children2tensor = torch.tensor(children, device=DEVICE, dtype=torch.long)
			label = torch.tensor(label, device=DEVICE)
			out = tbspp(nodes2vec, children2tensor)
			_, predicted = torch.max(out.data, 1)
			correct_list.append(label)
			predict_list.append(predicted)
	print('Accuracy:', accuracy_score(correct_list, predict_list))
	print(classification_report(correct_list, predict_list, target_names=list(labels)))
	print(confusion_matrix(correct_list, predict_list))
			# total += label.size(0)
			# correct += (predicted == label).sum().item()
	# print('Accuracy of the tework on the testing data set is: %.3f %%' % (100* correct/total))

if __name__ == '__main__':
	main()