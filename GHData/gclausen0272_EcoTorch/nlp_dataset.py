from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertModel, BertTokenizer
import numpy as np

class TextClassificationDataset(Dataset):
    def __init__(self, num_datapoints=1000, max_len=128, num_classes=10, tokenizer='kappa'):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.num_datapoints = num_datapoints
        self.max_len = 128
        self.num_classes = num_classes
        self.labels = torch.randint(low=0, high=self.num_classes, size=(1, self.num_datapoints))        
        self.words = ["wild", "west", "open", "oven", "apple", "orange", "hogwarts"]
        self.data = [' '.join(random.sample(self.words, 3)) for i in range(self.num_datapoints)]
        self.tokenized_data = [self.tokenizer(phrase, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt')
                                for phrase in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.tokenized_data[idx]
        labels = self.labels[0][idx]
        return text, labels
