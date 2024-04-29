import torch
import numpy as np

from bpemb import BPEmb

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self):
        self.loaded_file = self.load_file("./bangla_text_train.txt")
        self.data = self.preprocess_text(self.loaded_file)
        self.token_ids = self.tokenize(self.data)
        self.sequences = self.prepare_sequence(self.token_ids)
        self.sequences = np.array(self.sequences)
        self.X, self.y = self.prepare_data(self.sequences)
        self.y = self.to_categorical(self.y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return(
            torch.tensor(self.X[index]),
            torch.tensor(self.y[index])
        )
    
    def load_file(self, filepath: str):
        raw_text = open(filepath, 'r', encoding='utf-8').read()
        return raw_text
    
    def preprocess_text(self, text: str):
        data = text.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('“','').replace('”','')
        data = data.split()
        data = ' '.join(data)

        return data
        
    def tokenize(self, data: str):
        tokenizer = BPEmb(lang='bn', vs=5000, dim=300)
        token_ids = tokenizer.encode_ids(data)
        return token_ids
    
    def prepare_sequence(self,tokenized_data: list):
        sequences = []
        for i in range(3, len(tokenized_data)):
            words = tokenized_data[i-3:i+1]
            sequences.append(words)
        return sequences

    def prepare_data(self, sequences):
        X = []
        y = []

        for i in sequences:
            X.append(i[0:3])
            y.append(i[3])
            
        X = np.array(X)
        y = np.array(y)

        return X, y
        
    def to_categorical(self, y, num_classes = 5000):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]


class DatasetValid(torch.utils.data.Dataset):
    def __init__(self):
        self.loaded_file = self.load_file("./bangla_text_valid.txt")
        self.data = self.preprocess_text(self.loaded_file)
        self.token_ids = self.tokenize(self.data)
        self.sequences = self.prepare_sequence(self.token_ids)
        self.sequences = np.array(self.sequences)
        self.X, self.y = self.prepare_data(self.sequences)
        self.y = self.to_categorical(self.y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return(
            torch.tensor(self.X[index]),
            torch.tensor(self.y[index])
        )
    
    def load_file(self, filepath: str):
        raw_text = open(filepath, 'r', encoding='utf-8').read()
        return raw_text
    
    def preprocess_text(self, text: str):
        data = text.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('“','').replace('”','')
        data = data.split()
        data = ' '.join(data)

        return data
        
    def tokenize(self, data: str):
        tokenizer = BPEmb(lang='bn', vs=5000, dim=300)
        token_ids = tokenizer.encode_ids(data)
        return token_ids
    
    def prepare_sequence(self,tokenized_data: list):
        sequences = []
        for i in range(3, len(tokenized_data)):
            words = tokenized_data[i-3:i+1]
            sequences.append(words)
        return sequences

    def prepare_data(self, sequences):
        X = []
        y = []

        for i in sequences:
            X.append(i[0:3])
            y.append(i[3])
            
        X = np.array(X)
        y = np.array(y)

        return X, y
        
    def to_categorical(self, y, num_classes = 5000):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]

    
