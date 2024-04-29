# coding: utf8
import sys
import os
import collections
import random
from PIL import Image
import numpy as np
import pickle

import torchvision.transforms as transforms

class DataReader(object):
    def __init__(self, vocab_path, data_path, image_path, vocab_size=250000,
            batch_size=512, max_seq_len=32, is_shuffle=False):
        """ init
        """
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._max_seq_len = max_seq_len
        self._is_shuffle = is_shuffle
        self._data = self._build_data(data_path)
        self._image_path = image_path

        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
                ])

        if not os.path.exists(vocab_path):
            self._word_to_id = self._build_vocab(data_path)
            with open(vocab_path, "wb") as ofs:
                pickle.dump(self._word_to_id, ofs)
        else:
            with open(vocab_path, "rb") as ifs:
                self._word_to_id = pickle.load(ifs, encoding="iso-8859-1")

    def _build_vocab(self, filename):
        with open(filename, "r") as ifs:
            data = ifs.read().replace("\n", " ").replace("\t", " ").split()
        counter = collections.Counter(data)
        count_pairs = counter.most_common(self._vocab_size - 2)

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(2, len(words) + 2)))
        word_to_id["<pad>"] = 0
        word_to_id["<unk>"] = 1
        print("vocab words num: ", len(word_to_id))

    def _build_data(self, filename):
        with open(filename, "r") as ifs:
            lines = ifs.readlines()
            data = list(map(lambda x: x.strip().split("\t"), lines))
            if self._is_shuffle:
                random.shuffle(data)
        return data

    def _padding_batch(self, batch):
        # neg sample
        for idx, line in enumerate(batch[1]):
            neg_idx = random.randint(0, len(batch[1]) -1)
            while neg_idx == idx:
                neg_idx = random.randint(0, len(batch[1]) -1)
            batch[2].append(batch[1][neg_idx])
        batch[0] = np.asarray(batch[0])
        return batch

    def batch_generator(self):
        curr_size = 0
        batch = [[], [], []]
        for line in self._data:
            if len(line) != 2:
                continue
            curr_size += 1
            query, imageid = line
            query_list = query.split("\1\2")
            random.shuffle(query_list)
            query = query_list[0]
            query_ids = [self._word_to_id.get(x, self._word_to_id["<unk>"]) for x in query.split()]
            if len(query_ids) > self._max_seq_len:
                query_ids = query_ids[:self._max_seq_len]
            else:
                query_ids = query_ids + [self._word_to_id["<pad>"]] * (self._max_seq_len - len(query_ids))

            image = Image.open(self._image_path + "/%s.jpg" % imageid).convert('RGB')
            image = self.transform(image)

            batch[0].append(query_ids)
            batch[1].append(image)
            if curr_size >= self._batch_size:
                yield self._padding_batch(batch)
                batch = [[], [], []]
                curr_size = 0
        if curr_size > 0:
            yield self._padding_batch(batch)

    def extract_emb_generator(self):
        curr_size = 0
        batch = []
        for line in self._data:
            if len(line) != 1:
                continue
            curr_size += 1
            query = line[0]
            query_ids = [self._word_to_id.get(x, self._word_to_id["<unk>"]) for x in query.split()]
            query_ids = query_ids + [self._word_to_id["<pad>"]] * (self._max_seq_len - len(query_ids))
            batch.append(np.asarray(query_ids))
            if curr_size >= self._batch_size:
                yield np.asarray(batch)
                batch = []
                curr_size = 0
        if curr_size > 0:
            yield np.asarray(batch)

if __name__ == "__main__":
    reader = DataReader("data/vocab.pkl", "data/train.txt", "data/images")
    for batch in reader.batch_generator():
        for idx, line in enumerate(batch[0]):
            print(batch[0][idx], batch[1][idx], batch[2][idx])
            #print(batch[1][idx].shape, batch[2][idx].shape)
           
