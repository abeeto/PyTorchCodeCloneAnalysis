from collections import Counter
from pathlib import Path
import torch
from random import shuffle

dir_folder = Path('/Users/jungwon-c/Documents/ML Logistic/data')
positive = dir_folder/'books'/'positive.review'
negative = dir_folder/'books'/'negative.review'

UNK = '<UNK>'

# (token, freq) 형식으로 만들기
def make_datum(path: Path):
	with path.open(mode='r', encoding='utf-8') as review:
		datum = []
		for line in review:
			sentence = []
			for word in line.strip().split():
				divide = word.find(':')
				token = word[:divide]
				freq = int(word[divide+1:])
				sentence.append((token, freq))
			datum.append(sentence)
		return datum

# 사전 만들기
def make_vocabulary(vocab_size: int, counter: Counter):
	pos_datum = make_datum(positive)
	neg_datum = make_datum(negative)

	pos_tuples = [(token, freq) for sentence in pos_datum for token, freq in sentence]
	neg_tuples = [(token, freq) for sentence in neg_datum for token, freq in sentence]

	tuples = pos_tuples + neg_tuples

	for token, freq in tuples:
		counter[token] += freq

	vocab = {}
	for idx, (token, _) in enumerate(counter.most_common(vocab_size)):
		vocab[token] = idx
	vocab[UNK] = len(vocab)
	return vocab

# BoW 벡터 만들기
def make_BoW(vocab_size: int, sentence: list):
	vocab = make_vocabulary(vocab_size, counter)

	BoW = [0] * vocab_size
	for token, freq in sentence:
		if token in vocab.keys():
			BoW[vocab[token]] = 1
		else:
			BoW[vocab_size-1] = 1
	return BoW

# 타켓값 부여된 BoW벡터 만들기
def vec_with_target(path: Path, vocab_size: int, counter: Counter, target: float):
	datum = make_datum(path)
	vocab = make_vocabulary(vocab_size, counter)

	data_set = []
	for sentence in datum:
		BoW = make_BoW(vocab_size, sentence, vocab)
		data_set.append((BoW, target))
	return data_set

# 데이터 셔플
def make_train_data(vocab_size: int, counter:Counter):
	pos_datum = vec_with_target(positive, vocab_size, counter, 1.0)
	neg_datum = vec_with_target(negative, vocab_size, counter, 0.0)

	data_set = pos_datum + neg_datum
	shuffle(data_set)
	datum, target = zip(*data_set)
	return datum, target


if __name__ == '__main__':
	counter = Counter()
	make_vocabulary(300, counter)
