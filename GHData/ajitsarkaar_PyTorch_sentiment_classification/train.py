import os
from os.path import join as pjoin
import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
from torch.utils.data import Dataset, DataLoader

from dataset import IMDBDataset
from model import SentimentNet

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# Globals
HOME = os.path.expanduser('~')
course_path = pjoin(HOME, 'ece6524', 'p2') 

time_now = datetime.datetime.now()
log_filename = time_now.strftime("SentimentNet_%d%b%Y_%Hh%Mm%Ss")
output_path = pjoin(course_path, log_filename)
if not os.path.isdir(output_path):
	os.mkdir(output_path)

# Logger
import logging
logs_path = pjoin(output_path, 'LOGS_{}.txt'.format(log_filename))
head = '%(asctime)-15s | %(filename)-10s | line %(lineno)-3d: %(message)s'
logging.basicConfig(filename=logs_path, format=head)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(head))
logger.addHandler(console)

context = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
logger.info('COMPUTE DEVICE: {}'.format(context))
gpu_flag = (context != torch.device('cpu'))

def main(c):
	
	# Hyperparameters
	lr = c['lr']
	bs = c['bs']
	ep = c['ep']
	word_size = c['word_size']
	feature_dim = c['feature_dim']
	sq = c['sequence_length']

	imdb = IMDBDataset()
	tr_iter, va_iter = imdb.create_dataset(sq, bs)
	nbatches = len(tr_iter)
	recurr_unit = 'LSTM'
	nl = 1
	
	features = [1]
	fnames = ['1']
	current_feature = 'NUMBER OF LAYERS'

	plt.figure(figsize=(12, 8))

	for idx, feat in enumerate(features):
		logger.info('----------------------------------------------------------------')
		logger.info('++++++++                    PARAMS                      ++++++++')
		logger.info('UNIT           : {}'.format(recurr_unit))
		logger.info('# LAYERS       : {:1d}'.format(feat))
		logger.info('EMBEDDING SIZE : {:4d}'.format(feature_dim))
		logger.info('SEQUENCE LENGTH: {:4d}'.format(sq))
		logger.info('----------------------------------------------------------------')

		feat_name = fnames[idx]
		
		model = SentimentNet(unit=recurr_unit, 
							 num_layers=feat,
							 vocab_size=imdb.get_vocab_size(), 
							 word_size=word_size, 
							 sequence_length=sq, 
							 pad_idxs=imdb.get_pad_idx(), 
							 embedding_size=feature_dim, 
							 num_classes=2).to(context)

		val_perf = []
		epnums = []
		
		for en in range(1, ep + 1):
			logger.info('################################################################')
			logger.info('#### EPOCH # {:4d}'.format(en))
			logger.info('################################################################')
			
			trained_model = train(model, tr_iter, c)
			val_acc = validate(trained_model, va_iter, c)
			val_perf.append(val_acc)
			epnums.append(en)
		
		plt.plot(epnums, val_perf)
		mpath = save_model(trained_model, output_path, feat_name)

	plt.xlabel('EPOCHS')
	plt.ylabel('ACCURACY (%)')
	plt.title('Performance of {}'.format(current_feature))
	plt.legend(fnames)
	plt.savefig(pjoin(output_path, '{}_{}.png'.format(feat_name, 'plots')))
		
def train(model, dl, c):
	lr = c['lr']
	bs = c['bs']
	ep = c['ep']
	optimizer = O.Adam(model.parameters(), lr=c['lr'])
	criterion = nn.CrossEntropyLoss()
	log_freq = 5
	nb = len(dl)

	for bn, batch in enumerate(dl):
		text = batch.text
		labels = batch.label
		text = text.to(context) if gpu_flag else text
		labels = labels.to(context) if gpu_flag else labels
		
		if text.shape[-1] != bs:
			continue
		optimizer.zero_grad()
		logits, hidden_state = model(text)
		loss = criterion(logits, labels.long())
		ncorrect = ((torch.argmax(logits, dim=1) == (labels.long())).sum().item())
		
		if (bn + 1) % log_freq == 0:
			train_acc = ncorrect/bs
			logger.info('# BATCH: {:4d} | LOSS: {:9.4f} | TRAIN ACC: {:8.4f} %'.format(bn+1, 
																					   loss, 
																					   train_acc * 100))
		
		loss.backward()
		optimizer.step()
	
	return model

def validate(model, dl, c):
	bs = c['bs']
	ncorrect = 0
	ntotal = 0
	
	model.eval()
	
	logger.info('')
	logger.info('**** RUNNING VALIDATION...')

	with torch.no_grad():
		for bn, batch in enumerate(dl):
			text = batch.text
			labels = batch.label
			text = text.to(context) if gpu_flag else text
			labels = labels.to(context) if gpu_flag else labels
			if text.shape[-1] != bs:
				continue
			
			logits, hidden_state = model(text)
			ntotal += len(labels)
			ncorrect += ((torch.argmax(logits, dim=1) == (labels.long())).sum().item())
	
	logger.info('**** VALIDATION ACCURACY = {:5.2f} %'.format((ncorrect/ntotal) * 100))
	logger.info('')
	logger.info('')

	model.train()
	return (ncorrect/ntotal) * 100

def save_model(model, opdir, feat_name=''):
	time_now = datetime.datetime.now()
	model_name = time_now.strftime("%d%b%Y_%Hh%Mm%Ss")
	model_path = pjoin(opdir, '{}_{}.pt'.format(feat_name, model_name))
	torch.save(model.state_dict(), model_path)
	return model_path

if __name__ == '__main__':
	cfg = {
		'lr': 0.0001,
		'weight_decay': 0.005,
		'bs': 128,
		'ep': 25,
		'word_size': 300,
		'feature_dim': 64,
		'sequence_length': 500
	}
	
	main(cfg)