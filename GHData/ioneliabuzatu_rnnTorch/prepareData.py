from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string


class PrepareData():
    def __init__(self):
        self.all_letters = string.ascii_letters + " .,;'"
        self.n_letters = len(self.all_letters)
        # build the dictionary for the names
        self.category_lines = {}  # data: used for training and testing
        self.all_categories = []  # store all names of categories

    def findFiles(self, path):
        return glob.glob(path)

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    # Read a file and split into lines
    def readLines(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.unicodeToAscii(line) for line in lines]

    def outcome(self):
        for filename in self.findFiles("data/names/*.txt"):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = self.readLines(filename)
            self.category_lines[category] = lines

        return self.all_letters, self.n_letters,  self.category_lines, self.all_categories


if __name__ == '__main__':
    PrepareData()
