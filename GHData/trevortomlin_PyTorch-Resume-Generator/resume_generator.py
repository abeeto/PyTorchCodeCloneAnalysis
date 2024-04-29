import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from cleantext import clean
from os.path import exists
import pickle

# For visualizing data
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os

# Helpful Reference
# https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb

RESUME_PATH = 'data/Resume/Resume.csv'
CLEANED_TEXT_PATH = 'data/cleaned_text'
PRINTABLE_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ 	\n"
NUM_CHARS = len(PRINTABLE_CHARS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_text(text):

	for x in tqdm(range(len(text)), desc="Cleaning..."):

		text[x] = clean(text[x], lower=True, to_ascii=True, fix_unicode=True)

		text[x] = text[x].replace('\n', " ")
		text[x] = text[x].replace("city ,", "city,")
		text[x] = text[x].replace(" - ", ", ")
		text[x] = text[x].replace(" ,", ",")
		text[x] = text[x].replace(" :", ":")

		text[x] = ''.join([y for y in text[x] if y in PRINTABLE_CHARS])

	os.system('cls')

	print("Cleaning Complete.")

def get_text_from_csv(path):

	df = pd.read_csv(path, sep=',', dtype=str, encoding='utf-8', index_col=False)

	df = df['Resume_str']

	return df.values.tolist()

def write_cleaned_text(text, file):

	with open(file, 'wb') as f:
		pickle.dump(text, f)

def read_cleaned_text(file):

	with open(file, 'rb') as f:
		text = pickle.load(f)

	return text

class ResumeDataset(Dataset):

	def __init__(self, path):

		self.data = []
		self._load(path)

	def _load(self, path):

		cleaned_text_file = CLEANED_TEXT_PATH + "/cleaned_text.pickle"
		tensor_file = CLEANED_TEXT_PATH + "/tensor_data.pickle"

		# Saving text data 
		if not (exists(cleaned_text_file)):
			text = get_text_from_csv(path)
			clean_text(text)
			write_cleaned_text(text, cleaned_text_file)

		cleaned_data = read_cleaned_text(cleaned_text_file)

		if not (exists(tensor_file)):

			# Converting text data to one hot encoded tensor
			for resume in cleaned_data:
				tensor = torch.zeros(len(resume)).long()

				for c in range(len(resume)):
					try:
						tensor[c] = PRINTABLE_CHARS.index(resume[c])
					except(ValueError):
						print(resume[c])

				self.data.append(tensor)

			write_cleaned_text(self.data, tensor_file)

		self.data = read_cleaned_text(tensor_file)

		#lenData = [len(x) for x in self.data]

		#dataGT5k = [x for x in self.data if len(x) >= 3500]
		#print(len(dataGT5k))

		# plt.hist(lenData)
		# plt.axis([0, 40000, 0, 2000])
		# plt.xlabel("Number of Characters")
		# plt.ylabel("Number of Resumes")
		# plt.show()

		# Stats for length of data

		#avgLen = sum(map(len, self.data))/float(len(self.data))

		# self.data.sort()
		# mid = len(self.data) // 2
		# medLen = (len(self.data[mid]) + len(self.data[~mid])) / 2

		# maxLen = 0
		# minLen = 100000

		# for x in self.data:
		# 	maxLen = max(maxLen, len(x))
		# 	minLen = min(minLen, len(x))

		# print("AvgLen: ", avgLen)
		# print("MaxLen: ", maxLen)
		# print("MinLen: ", minLen)
		# print("MedLen: ", medLen)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		return self.data[i]

class RNN(nn.Module):

	def __init__(self, input_size, hidden_size, output_size, n_layers=1):
		super(RNN, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers

		self.encoder = nn.Embedding(input_size, hidden_size)
		self.rnn = nn.RNN(hidden_size, hidden_size, n_layers)
		self.decoder = nn.Linear(hidden_size, output_size)

	def forward(self, input_t, hidden_t):
		input_t = self.encoder(input_t.view(1, -1))
		output_t, hidden_t = self.rnn(input_t.view(1, 1, -1), hidden_t)
		output_t = self.decoder(output_t.view(1, -1))
		return output_t, hidden_t

	def init_hidden(self):
		return torch.zeros(self.n_layers, 1, self.hidden_size)

def main():

	dataset = ResumeDataset(RESUME_PATH)
	dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

	print(len(dataset.data))

	torch.set_printoptions(threshold=10000)

	print(dataset.data[0])

	#print(dataset[0])

	#print(PRINTABLE_CHARS)

	#print(NUM_CHARS)

	# FOR TRAINING
	# NEEDS TO BE MODIFIED LATER

	# rnn = RNN(64, 64, 64)

	# input_tensor = torch.zeros(1, 1, 1)
	# hidden_tensor = rnn.init_hidden()

	# output, next_hidden = rnn(input_tensor, hidden_tensor)


if __name__ == '__main__':
	main()