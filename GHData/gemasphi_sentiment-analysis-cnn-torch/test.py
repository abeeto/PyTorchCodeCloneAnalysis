from readers import StandfordDataSetReader, GloveReader
from model import Net 
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def test(data_info, model_arch, training_params, log_interval):
	glove = GloveReader(path =  data_info['GLOVE'])
	embeddings, word2id = glove.get()


	sd = StandfordDataSetReader(path = data_info['SD'])
	data, labels = sd.get("train", word2id)
	data = torch.utils.data.DataLoader(data, batch_size = training_params['BATCH_SIZE'])


	model = Net(model_arch, vocab_size = len(word2id), embedding_dim = training_params['EMBEDDING_DIM'], use_embeddings = training_params["USE_EMBEDDINGS"], embeddings = embeddings)
	model.load_state_dict(data_info["MODEL_NAME"])


	with torch.no_grad():
		model.eval()
		loss = 0
		correct = 0
		for i, batch in enumerate(data):
			if len(labels) < i*training_params['BATCH_SIZE'] + training_params['BATCH_SIZE']:
				break

			batch_labels = labels[i*training_params['BATCH_SIZE'] : training_params['BATCH_SIZE']*(i + 1)]
			batch, batch_labels = batch.to(training_params['DEVICE']), batch_labels.to(training_params['DEVICE'])
			batch_labels = batch_labels.view((training_params['BATCH_SIZE'],))

			output = model(batch.long())
			loss += F.cross_entropy(output, batch_labels)

			pred = output.argmax(dim=1, keepdim=True)
			correct += pred.eq(batch_labels.view_as(pred)).sum().item()

			if i % log_interval == 0:
				print('[{}/{} ({:.0f}%)]'.format(i * training_params['BATCH_SIZE'], len(data.dataset),
	                               100*i*training_params['BATCH_SIZE'] / len(data.dataset)))

	
	loss /= len(data.dataset)
	print(loss)
	print(correct / len(data.dataset))
	return loss, correct / len(data.dataset)
