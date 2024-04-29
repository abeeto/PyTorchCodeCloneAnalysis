import pandas as pd
import pytreebank
import torch
import pickle
import os 
import numpy as np

class StandfordDataSetReader(object):
	def __init__(self, path = "asd", labels_path = "labels.pt", data_path = "train.pt", word2id_path = "standford_word2id.pt"):
		self.labels_path = labels_path
		self.data_path = data_path
		self.path = path
		self.word2id_path = word2id_path
		
	def process(self, type, word2id):
		data = pytreebank.load_sst(self.path)
		train, labels, max_sentence_size = self._phrase2id(data[type], word2id) 
		t_data, t_labels = self._create_torch_training(train, labels, max_sentence_size)

		torch.save(t_data, self.data_path)
		torch.save(t_labels, self.labels_path)

		return t_data, t_labels

	def import_data(self):
		t_data = torch.load(self.data_path)
		t_labels = torch.load(self.labels_path)
		t_labels = torch.unsqueeze(t_labels, 1)

		return t_data, t_labels

	def get(self, type, word2id):
		alread_processed = os.path.isfile(self.labels_path) \
						and os.path.isfile(self.data_path) \

		if(alread_processed):
			return self.import_data()
		else:
			return self.process(type, word2id)

	def build_vocab(self, cut_off): #this is basically useless, since glove already has the most common words
		word_count = {}
		data = pytreebank.load_sst(self.path)

		for phrase in data['train']:
			phrase.lowercase()
			_, sentence = phrase.to_labeled_lines()[0]
			
			for word in sentence.split():
				#check if stop word and ignore
				if word in word_count:
					word_count[word] += 1
				else:
					word_count[word] = 1

		filter_word_count = [word for word, count in word_count.items() if count < cut_off]
		word2id = {word: i + 1  for i, word in enumerate(filter_word_count)}

		pickle.dump(word2id, open(self.word2id_path, 'wb'))

		return word2id

	def _phrase2id(self, data, word2id):
		train = []
		labels = []
		max_sentence_size = 0

		for phrase in data:
			phrase.lowercase()
			label, sentence = phrase.to_labeled_lines()[0]
			new_sentence = self._sentence2id(sentence, word2id) 

			n_s_size = len(new_sentence)
			if n_s_size != 0:
				train.append(new_sentence)
				labels.append(label)

				if n_s_size > max_sentence_size:
					max_sentence_size = n_s_size

		return train, labels, max_sentence_size


	def _sentence2id(self, sentence, word2id):
		new_sentence = []
		for word in sentence.split():
			if word in word2id:
				new_sentence.append(word2id[word])

		return new_sentence


	def _create_torch_training(self, train, labels, max_sentence_size):
		t_data = torch.zeros(len(train), max_sentence_size, dtype=torch.int32)

		for i, data in enumerate(train):
			t_data[i] = torch.Tensor((data + max_sentence_size*[0])[:max_sentence_size] )

		return t_data, torch.tensor(labels)



class GloveReader(object):
	def __init__(self, path = "glove.6B.300d.txt",  output_path = "embeddings.pkl", word2id = "glove_word2id.pkl"):
		self.path = path
		self.output_path = output_path
		self.word2id = word2id

	def process(self):
		embeddings = []
		word2id = {}

		with open(self.path, 'rb') as f:
			i = 1
			for line in f:
				line = line.decode().split()
				word2id[line[0]] = i
				embeddings.append(np.array(line[1:]).astype(np.float)) 
				i += 1
		
		t_embeddings = torch.zeros((i - 1, 300))
		for j in range(i - 1):
			t_embeddings[j] =  torch.from_numpy(embeddings[j])

		torch.save(t_embeddings, self.output_path)
		pickle.dump(word2id, open(self.word2id, 'wb'))

		return embeddings, word2id  

	def import_data(self):
		embeddings = torch.load(self.output_path)
		word2id = pickle.load(open(self.word2id, 'rb'))

		return embeddings, word2id  

	def get(self):
		alread_processed = os.path.isfile(self.output_path) \
						and os.path.isfile(self.word2id)

		if(alread_processed):
			return self.import_data()
		else:
			return self.process()

