# -*- coding: utf-8 -*- #

""" names dataset
"""

from io import open
import glob, os, unicodedata, string
import random

import torch


def find_files(path):
    return glob.glob(path)


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


class Dataset:

    def __init__(self):
        self.all_letters = string.ascii_letters + " .,;'-"
        self.n_letters = len(self.all_letters)
        self.category_lines = {}
        self.all_categories = []
        self.n_categories = None

        for filename in find_files('names_data/data/names/*.txt'):
            category, _ = os.path.splitext(os.path.basename(filename))
            self.all_categories.append(category)
            lines = self.read_lines(filename)
            self.category_lines[category] = lines

        self.n_categories = len(self.all_categories)

        if self.n_categories == 0:
            raise RuntimeError('Data not found.')

        print(f'# categories: {self.n_categories}:: {self.all_categories}')

    def unicode2ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    def read_lines(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split("\n")
        return [self.unicode2ascii(line) for line in lines]

    def random_training_pair(self):
        category = random_choice(self.all_categories)
        line = random_choice(self.category_lines[category])
        return category, line

    def category_tensor(self, category):
        li = self.all_categories.index(category)
        tensor = torch.zeros(1, self.n_categories)
        tensor[0][li] = 1
        return tensor

    def input_tensor(self, line):
        """ one hot matrix of first to last letters (not including EOS) """
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li in range(len(line)):
            letter = line[li]
            tensor[li][0][self.all_letters.find(letter)] = 1
        return tensor

    def target_tensor(self, line):
        """ long tensor of second letter to EOS """
        letter_indices = [self.all_letters.find(line[li]) for li in range(1, len(line))]
        letter_indices.append(self.n_letters - 1)
        return torch.LongTensor(letter_indices)

    def random_training_example(self):
        category, line = self.random_training_pair()
        category_tensor_var = self.category_tensor(category)
        input_line_tensor = self.input_tensor(line)
        target_line_tensor = self.target_tensor(line)
        return category_tensor_var, input_line_tensor, target_line_tensor
