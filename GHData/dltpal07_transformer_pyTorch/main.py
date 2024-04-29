import os
import pdb
import csv
import random
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import time, datetime
import argparse
import numpy as np
from pathlib import Path

import utils, dataloader
from model.transformer import Transformer
from torch.optim import SGD, Adam
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='NMT - Transformer')
""" recommend to use default settings """

# environmental settings
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save', action='store_true', default=0)

# architecture
parser.add_argument('--num_enc_layers', type=int, default=6, help='Number of Encoder layers')
parser.add_argument('--num_dec_layers', type=int, default=6, help='Number of Decoder layers')
parser.add_argument('--num_token', type=int, help='Number of Tokens')
parser.add_argument('--max_len', type=int, default=20)
parser.add_argument('--model_dim', type=int, default=512, help='Dimension size of model dimension')
parser.add_argument('--hidden_size', type=int, default=2048, help='Dimension size of hidden states')
parser.add_argument('--d_k', type=int, default=64, help='Dimension size of Key and Query')
parser.add_argument('--d_v', type=int, default=64, help='Dimension size of Value')
parser.add_argument('--n_head', type=int, default=8, help='Number of multi-head Attention')
parser.add_argument('--d_prob', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--max_norm', type=float, default=5.0)

# hyper-parameters
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 hyper-parameter for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.98, help='Beta2 hyper-parameter for Adam optimizer')
parser.add_argument('--eps', type=float, default=1e-9, help='Epsilon hyper-parameter for Adam optimizer')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--teacher-forcing', action='store_true', default=False)
parser.add_argument('--warmup_steps', type=int, default=78, help='Warmup step for scheduler')
parser.add_argument('--logging_steps', type=int, default=500, help='Logging step for tensorboard')
# etc
parser.add_argument('--k', type=int, default=4, help='hyper-paramter for BLEU score')

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else 'cpu')


def train(dataloader, epochs, model, criterion, args, vocab, i2w):
	model.train()
	model.zero_grad()
	optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.eps)
	correct = 0

	cnt = 0
	total_score = 0.
	global_step = 0
	tr_loss = 0.
	for epoch in range(epochs):

		for idx, (src, tgt) in enumerate(dataloader):
			src, tgt = src.to(device), tgt.to(device)
			"""
			ToDo: feed the input to model
			src.size() : [batch, max length]
			tgt.size() : [batch, max lenght + 1], the first token is always [SOS] token.
			
			These codes are one of the example to train model, so changing codes are acceptable.
			But when you change it, please left comments.
			If model is run by your code and works well, you will get points.
			"""
			optimizer.zero_grad()

			outputs = model(src, tgt[:, :-1])
			tgt = tgt[:, 1:]

			tgt = torch.flatten(tgt)
			outputs = outputs.reshape(len(tgt), -1)
			loss = criterion(outputs, tgt)
			tr_loss += loss.item()

			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
			optimizer.step()
			global_step += 1

			pred = outputs.argmax(dim=1, keepdim=True)
			pred_acc = pred[tgt != 2]
			tgt_acc = tgt[tgt != 2]

			correct += pred_acc.eq(tgt_acc.view_as(pred_acc)).sum().item()

			cnt += tgt_acc.shape[0]
			score = 0.

			# model.eval()
			with torch.no_grad():
				pred = pred.reshape(args.batch_size, args.max_len, -1).detach().cpu().tolist()
				tgt = tgt.reshape(args.batch_size, args.max_len).detach().cpu().tolist()
				for p, t in zip(pred, tgt):
					eos_idx = t.index(vocab['[PAD]']) if vocab['[PAD]'] in t else len(t)
					p_seq = [i2w[i[0]] for i in p][:eos_idx]
					t_seq = [i2w[i] for i in t][:eos_idx]
					k = args.k if len(t_seq) > args.k else len(t_seq)
					s = utils.bleu_score(p_seq, t_seq, k=k)
					score += s
					total_score += s

			score /= args.batch_size

			# verbose
			batches_done = (epoch - 1) * len(dataloader) + idx
			batches_left = args.n_epochs * len(dataloader) - batches_done
			prev_time = time.time()
			print("\r[epoch {:3d}/{:3d}] [batch {:4d}/{:4d}] loss: {:.6f} acc: {:.4f} BLEU: {:.4f})".format(
				epoch, args.n_epochs, idx + 1, len(dataloader), loss, correct / cnt, score), end=' ')

	tr_loss /= cnt
	tr_acc = correct / cnt
	tr_score = total_score / len(dataloader.dataset) / epochs

	return tr_loss, tr_acc, tr_score

def eval(dataloader, model, args, lengths=None):
	model.eval()

	idx = 0
	total_pred = []
	total = 0
	with torch.no_grad():
		correct = 0.
		cnt = 0.
		for src, tgt in dataloader:
			src, tgt = src.to(device), tgt.to(device)
			"""
			ToDo: feed the input to model
			src.size() : [batch, max length]
			tgt.size() : [batch, max lenght + 1], the first token is always [SOS] token.

			These codes are one of the example to train model, so changing codes are acceptable.
			If model is run by your code and works well, you will get points.
			"""
			outputs = model(src, tgt, False)
			for i in range(outputs.shape[0]):
				total_pred.append(torch.argmax(outputs[i, :lengths[idx]], dim=-1).cpu())
				idx += 1
				"""
				ToDo: Output (total_pred) is the model predict of test dataset
				Variable lenghts is the length information of the target length.
				"""
	print(total)
	total_pred = np.concatenate(total_pred)
	return total_pred

def main():


	utils.set_random_seed(args)
	tr_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
											 src_filepath='./data/de-en/nmt_simple.src.train.txt',
											 tgt_filepath='./data/de-en/nmt_simple.tgt.train.txt',
											 vocab=(None, None),
											 is_src=True, is_tgt=False, is_train=True)
	ts_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
											 src_filepath='./data/de-en/nmt_simple.src.test.txt',
											 tgt_filepath=None,
											 vocab=(tr_dataset.vocab, None),
											 is_src=True, is_tgt=False, is_train=False)


	vocab = tr_dataset.vocab
	i2w = {v: k for k, v in vocab.items()}
	tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
	ts_dataloader = DataLoader(ts_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2)
	args.num_token = len(tr_dataset.vocab)

	model = Transformer(num_token=args.num_token,
						max_seq_len=args.max_len,
						dim_model=args.model_dim,
						d_k=args.d_k,
						d_v=args.d_v,
						n_head=args.n_head,
						dim_hidden=args.hidden_size,
						d_prob=args.d_prob,
						n_enc_layer=args.num_enc_layers,
						n_dec_layer=args.num_dec_layers)

	model.init_weights()
	model = model.to(device)
	criterion = nn.NLLLoss(ignore_index=vocab['[PAD]'])

	tr_loss, tr_acc, tr_score = train(tr_dataloader, args.n_epochs, model, criterion, args, vocab, i2w)
	print("tr: ({:.4f}, {:5.2f}, {:5.2f}) | ".format(tr_loss, tr_acc * 100, tr_score * 100), end='')

	# for kaggle
	with open('./data/de-en/length.npy', 'rb') as f:
		lengths = np.load(f)
	pred = eval(ts_dataloader, model=model, args=args, lengths=lengths)
	pred_filepath = './data/pred_real_test.npy'
	np.save(pred_filepath, np.array(pred))

if __name__ == "__main__":
	main()
















