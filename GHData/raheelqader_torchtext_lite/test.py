import unittest
from torchtext_lite import *
# import torchtext
import sys
import math


src_path = './source.txt'
trg_path = './target.txt' 
src_max_length=200
trg_max_length=200

from time import time
last_time = time()
def cal_time(msg):
	global last_time
	current_time = time()
	seconds_elapsed = current_time - last_time
	print(msg, ' seconds_elapsed: ', seconds_elapsed)
	last_time = time()



def convert_size(size_bytes):
	if size_bytes == 0:
		 return "0B"
	size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
	i = int(math.floor(math.log(size_bytes, 1024)))
	p = math.pow(1024, i)
	s = round(size_bytes / p, 2)
	return "%s %s" % (s, size_name[i])


def tokenizer(text):
	return text.split('men')

class TestAll(unittest.TestCase):

	def test1(self):
		src_field = TextField()
		trg_field = TextField()
		src_raw_field = RawField()

		dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
		src_field.build_vocab([dataset.src,dataset.trg])
		dataset.save('e2e_dataset')


	def test2(self):
		dataset = Dataset()
		dataset.load('e2e_dataset')


	def test3(self):
		dataset = Dataset()
		dataset.load('e2e_dataset', )
		# dataset.build_vocab(vocab_size=50, share_vocab=True, src_vocab_path='vocab_file', save=True)

		print(sorted(dataset.fields['src'].vocab.word2index.items(), key= lambda x:x[1]))

	def test4(self):
		src_field = TextField(add_sos=True, add_eos=True)
		trg_field = TextField()
		src_raw_field = RawField()

		dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
		src_field.build_vocab([dataset.src,dataset.trg], vocab_size=50000)
		dataset.save('e2e_dataset')

		# src_field.process(dataset[:2])
		
		src_examples = []
		counter = 0
		for x in dataset.src:
			src_examples.append(x)
			counter+=1
			if counter>=4: break
		

		src_field.process(src_examples, None)


	def test5(self):
		src_field = TextField(add_sos=True, add_eos=True)
		trg_field = TextField()
		src_raw_field = RawField()

		dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
		src_field.build_vocab([dataset.src, dataset.trg], vocab_size=50000)
		dataset.save('e2e_dataset')

		# src_field.process(dataset[:2])
		
		src_examples = []
		counter = 0
		for x in dataset.src:
			src_examples.append(x)
			counter+=1
			if counter>=100: break
		

		print(src_field.process(src_examples, None))



	def test6(self):

		src_field = TextField()
		trg_field = TextField()
		fields = [('src', src_field), ('trg', trg_field)]

		dataset = Dataset()
		dataset.load('e2e_dataset')

		src_field.build_vocab([dataset.src, dataset.trg], vocab_size=50000)
		src_field.save('src_field')

		src_examples = []
		counter = 0
		for x in dataset.src:
			src_examples.append(x)
			counter+=1
			if counter>=100: break
		

		# print(src_field.process(src_examples))



	def test7(self):

		src_field = TextField()
		trg_field = TextField()
		fields = [('src', src_field), ('trg', trg_field)]

		dataset = Dataset()
		dataset.load('e2e_dataset')

		src_field.load('src_field')

		src_examples = []
		counter = 0
		for x in dataset.src:
			src_examples.append(x)
			counter+=1
			if counter>=4: break
		

		print(src_field.process(src_examples, None))



	def test8(self):
		src_field = TextField(add_sos=True, add_eos=True)
		trg_field = TextField()
		src_raw_field = RawField()

		dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
		src_field.build_vocab([dataset.src, dataset.trg], vocab_size=50000)


		src_examples = []
		counter = 0
		for x in dataset.src:
			src_examples.append(x)
			counter+=1
			if counter>=12: break
		
		print(src_field.process(src_examples, None))



	def test9(self):


			
		src_field = TextField(add_sos=True, add_eos=True, tokenizer=tokenizer)
		trg_field = TextField()
		src_raw_field = RawField()

		dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
		src_field.build_vocab([dataset.src, dataset.trg], vocab_size=50000)
		src_field.save('src_field')

		src_examples = []
		counter = 0
		for x in dataset.src:
			print(x)
			src_examples.append(x)
			counter+=1
			if counter>=12: break
		
		print(src_field.process(src_examples, None))



	def test10(self):

		src_field = TextField(add_sos=True, add_eos=True)
		trg_field = TextField()
		src_raw_field = RawField()

		dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
		src_field.build_vocab([dataset.src], vocab_size=50000)
		trg_field.build_vocab([dataset.trg], vocab_size=50000)

		src_examples = []
		counter = 0
		for x in dataset.trg:
			src_examples.append(x)
			counter+=1
			if counter>=12: break
		
		print(trg_field.process(src_examples, None))


	def test11(self):

		src_field = TextField(add_sos=True, add_eos=True)
		trg_field = TextField()
		src_raw_field = RawField()

		dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
		src_field.build_vocab([dataset.src], vocab_size=50000)
		trg_field.build_vocab([dataset.trg], vocab_size=50000)

		src_raw_examples = []
		counter = 0
		for x in dataset.src_raw:
			src_raw_examples.append(x)
			counter+=1
			if counter>=2: break
			print('*'*5)
		
		print(src_raw_field.process(src_raw_examples, None))




	def test12(self):
		src_field = TextField(add_sos=True, add_eos=True)
		trg_field = TextField()
		src_raw_field = RawField()

		dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
		src_field.build_vocab([dataset.src, dataset.trg], vocab_size=50000)

		dataset.save('dataset')
		src_field.save('src_field')
		trg_field.save('trg_field')
		src_raw_field.save('src_raw_field')


		src_examples = []
		counter = 0
		for x in dataset.src:
			src_examples.append(x)
			counter+=1
			if counter>=12: break
		
		print(src_field.process(src_examples, None))


	def test13(self):
		src_field = TextField(add_sos=True, add_eos=True)
		trg_field = TextField()
		src_raw_field = RawField()

		fields = [('src', src_field), ('trg', trg_field), ('src_raw', src_raw_field)]

		dataset = Dataset()
		dataset.load('e2e_dataset')

		src_field.load('src_field')
		# trg_field.load('trg_field')
		# src_raw_field.load('src_raw_field')

		src_raw_examples = []
		counter = 0
		for x in dataset.src_raw:
			src_raw_examples.append(x)
			counter+=1
			if counter>=2: break
			print('*'*5)
		
		print(src_raw_field.process(src_raw_examples, None))



	def test14(self):

		src_field = TextField(add_sos=True, add_eos=True)
		trg_field = TextField()
		src_raw_field = RawField()

		dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
		src_field.build_vocab([dataset.src], vocab_size=50000)
		trg_field.build_vocab([dataset.trg], vocab_size=50000)


		minibatch = []
		counter = 0
		for x in dataset:
			minibatch.append(x)
			counter+=1
			if counter>=2: break


		batch = Batch(minibatch, dataset.fields, None)
		print(batch.src_raw)
		print(len(batch))


	def test15(self):

		src_field = TextField(add_sos=True, add_eos=True)
		trg_field = TextField()
		src_raw_field = RawField()

		dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
		src_field.build_vocab([dataset.src, dataset.trg], vocab_size=50000)
		trg_field.build_vocab([dataset.trg], vocab_size=50000)

		batch_iterator = BucketIterator(dataset=dataset, batch_size=20, sort=False, shuffle=True, 
										sort_within_batch=True,  sort_key=lambda x: len(x.src), repeat=False)

		for batch in batch_iterator:

			# batch = batch.sort('trg')

			src, src_len = batch.src
			trg, trg_len = batch.trg
			print('batch', batch)

			print('src_len', src_len)
			print('trg_len', trg_len)


	def test16(self):


		last_time = time()
		

		src_field = TextField(add_sos=True, add_eos=True)
		trg_field = TextField()
		src_raw_field = RawField()

		dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
		src_field.build_vocab([dataset.src, dataset.trg], vocab_size=50000)
		trg_field.build_vocab([dataset.trg], vocab_size=50000)

		# bi = torchtext.BucketIterator(dataset, batch_size=100, sort_key=lambda x: len(x.src), sort=False, shuffle=True, repeat=True)
		bi = BucketIterator(dataset, batch_size=5, sort_key=lambda x: len(x.src), sort=False, shuffle=True, repeat=False, num_buckets=100)

		bg = bi.__iter__()

		for iter in range(0, 10):
			batch = next(bg)

			# print('bi: ', bi.epoch)
			# print('epoch_iter: ', bi.epoch_iterations)

			# batch = batch.sort('trg')

			src, src_len = batch.src
			trg, trg_len = batch.trg

			print('src_len', src_len, '	trg_len', trg_len)
			print('src_len', (src), '	trg_len', (trg))

		cal_time('BucketIterator')

		'''
		for batch in bi:

			print('bi: ', bi.epoch)
			print('epoch_iter: ', bi.epoch_iterations)

			# batch = batch.sort('trg')

			src, src_len = batch.src
			trg, trg_len = batch.trg

			print('src_len', src_len, '	trg_len', trg_len)
		'''



	def test17(self):

		last_time = time()

		src_field = TextField(add_sos=True, add_eos=True)
		trg_field = TextField()
		src_raw_field = RawField()

		dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
		src_field.build_vocab([dataset.src], vocab_size=50000)
		trg_field.build_vocab(dataset.trg, vocab_size=50000)

		bi = BucketIterator(dataset, batch_size=10, sort_key=lambda x: len(x.src), sort=False, shuffle=True, repeat=False, sort_within_batch=False)

		bg = bi.__iter__()

		for iter in range(0, 1):
			batch = next(bg)

			src, src_len = batch.src
			trg, trg_len = batch.trg

			print('src_len', src_len, '	trg_len', trg_len)
			
			batch.sort(sort_key=lambda x: len(x.src) , reverse=True)
			src, src_len = batch.src
			trg, trg_len = batch.trg
			
			print('src_len', src_len, '	trg_len', trg_len)
			print('src', src, '	trg', trg)



		cal_time('BucketIterator')


	def test18(self):

		""" test size of object in memorty """

		import os
		import psutil
		process = psutil.Process(os.getpid())

		last_time = time()

		src_field = TextField(add_sos=True, add_eos=True)
		trg_field = TextField()
		src_raw_field = RawField()

		dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
		src_field.build_vocab([dataset.src], vocab_size=50000, save_path='src_vocab')
		trg_field.build_vocab(dataset.trg, vocab_size=50000)

		bi = BucketIterator(dataset, batch_size=10, sort_key=lambda x: len(x.src), \
							sort=False, shuffle=False, repeat=False, sort_within_batch=False, num_buckets=1)

		for batch in bi:

			src_raws = batch.src_raw

			for src_raw in src_raws:
				print(src_raw)

		cal_time('nosort')
		print(convert_size(process.memory_info().rss))  # in bytes 


	def test19(self):

		""" test loading testset by usning saved field of train set without loading trainingset """

		# train_dataset = Dataset()
		# train_dataset.load('e2e_dataset')
		# train_dataset.fields['src'].save('src_field')
		# src_field = train_dataset.fields['src']

		src_field = TextField(add_sos=True, add_eos=True)
		src_field.load('src_field')
		fields = [('src', src_field)]
		
		batch_size = 100
		examples = []

		with open(src_path, encoding='utf-8') as src_file:
			for line_i, src_line in enumerate(src_file):
				src = tokenizer(src_line.strip())
				print('Reading line #: ', line_i, end='\r')

				if src_max_length is not None:
					src = src[:src_max_length]
					src = detokenizer(src)

				if src != '':
					examples.append(Example([src], fields))

					if len(examples) == batch_size:
						batch = Batch(examples, dict(fields), None)
						src, src_len = batch.src
						print('src', src.size())
						examples = []
		if examples:
			batch = Batch(examples, dict(fields), None)
			src, src_len = batch.src
			print('last: src', src.size())


	def test20(self):
		""" test loading testset by usning vocab of train set without loading trainingset """

		# src_field = TextField(add_sos=True, add_eos=True)
		# trg_field = TextField()
		# src_raw_field = RawField()
		# dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
		# src_field.build_vocab([dataset.src], save_path='src_vocab')

		src_field = TextField(add_sos=True, add_eos=True)
		src_raw_field = RawField()
		src_field.build_vocab(load_path='src_vocab')
		fields = [('src', src_field), ('src_raw', src_raw_field)]
		
		batch_size = 100
		examples = []

		with open(src_path, encoding='utf-8') as src_file:
			for line_i, src_line in enumerate(src_file):
				src = tokenizer(src_line.strip())
				print('Reading line #: ', line_i, end='\r')

				if src_max_length is not None:
					src = src[:src_max_length]
					src = detokenizer(src)

				if src != '':
					examples.append(Example([src, src], fields))

					if len(examples) == batch_size:
						batch = Batch(examples, dict(fields), None)
						src, src_len = batch.src
						print('src', src.size())
						examples = []
		if examples:
			batch = Batch(examples, dict(fields), None)
			src, src_len = batch.src
			print('last: src', src.size())

	def test21(self):
		""" test loading embedding """

		src_field = TextField(add_sos=True, add_eos=True)
		trg_field = TextField()
		src_raw_field = RawField()

		dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
		src_field.build_vocab([dataset.src], vocab_size=50000)
		trg_field.build_vocab(dataset.trg, vocab_size=50000)

		trg_field.vocab.load_embeddings('glove.6B.50d.txt')

		print(trg_field.vocab.embeddings.size())



	def test22(self):
		""" test bert field"""

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


		src_field = BertField(device)
		trg_field = TextField()
		src_raw_field = RawField()

		dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
		trg_field.build_vocab(dataset.trg, vocab_size=50000)

		bi = BucketIterator(dataset, batch_size=4, sort_key=lambda x: len(x.src), \
							sort=False, shuffle=False, repeat=False, sort_within_batch=False, num_buckets=1, device=device)

		for batch in bi:

			src, _ = batch.src
			print(src.size())
			for src_seq in src:
				print(src_seq)


if __name__ == '__main__':
	unittest.main()
	# test = TestAll()
	# test.test21()
