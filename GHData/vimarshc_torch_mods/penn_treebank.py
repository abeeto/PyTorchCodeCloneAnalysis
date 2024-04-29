import string
import csv
import os.path as op
import re
import torch
import codecs
import json
from torch.utils.data import DataLoader, Dataset
import torch.autograd as autograd

class PennTree(Dataset):
    def __init__(self, train_split=0.90, seqlen=50):
        """Create PenTree Bank dataset object.
        Arguments:
            l0: max length of a sample.
            train_split: % split of number of sentences
        """
        self.label_data_path = label_data_path
        # read alphabet
        # with open(alphabet_path) as alphabet_file:
        #     alphabet = str(''.join(json.load(alphabet_file)))
        


        self.alphabet = string.printable
        self.seqlen = seqlen
        # self.l0 = l0
        self.load(lowercase=False)
        # self.y = torch.LongTensor(self.label)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.oneHotEncode(idx)


    def load(self, lowercase=True):
        self.target = []
        self.data = []

        #Any repurcissions? 
        _data = ' '.join(nltk.corpus.treebank.words())
        _data_len = len(data)
        num_batches = _data_len//self.seqlen

        for i in range(num_batches):
            #Creating dataset in provided seq length
            self.data.append(_data[i*self.seqlen:(i+1)*self.seqlen])

            #Creating Respective targets.
            #Last character will have no target. 
            self.target.append(_data[i*self.seqlen + 1:(i+1)*self.seqlen + 1])

        # with open(self.label_data_path, 'r') as f:
        #     rdr = csv.reader(f, delimiter=',', quotechar='"')
        #     # num_samples = sum(1 for row in rdr)
        #     for index, row in enumerate(rdr):
        #         self.label.append(int(row[0]))
        #         txt = ' '.join(row[1:])
        #         if lowercase:
        #             txt = txt.lower()                
        #         self.data.append(txt)

    def oneHotEncode(self, idx):

        X = torch.zeros(self.seqlen, len(self.alphabet))
        y = torch.zeros(self.seqlen, len(self.alphabet))

        input_sequence = self.data[idx]
        target_sequence = self.target[idx]
        
        for _var, sequence in ((X,input_sequence), (y,target_sequence))
            for index_char, char in enumerate(sequence[::-1]):
                if self.char2Index(char)!=-1:
                    _var[index_char][self.char2Index(char)] = 1.0
        return X,y

    def char2Index(self, character):
        return self.alphabet.find(character)

    # def get_class_weight(self):
    #     num_samples = self.__len__()
    #     label_set = set(self.label)
    #     num_class = [self.label.count(c) for c in label_set]
    #     class_weight = [num_samples/float(self.label.count(c)) for c in label_set]    
    #     return class_weight, num_class
