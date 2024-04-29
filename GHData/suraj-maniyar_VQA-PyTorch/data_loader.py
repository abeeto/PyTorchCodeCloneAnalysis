from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from data_utils import preprocess_text, change
import numpy as np
import random


class ImageFeatureDataset(Dataset):
    def __init__(self, config, image_features, input_embedding, word2idx, word_list):
    
        self.config = config
        self.input_embedding = input_embedding
        self.word2idx = word2idx
        self.image_features = image_features
        self.word_list = word_list

    
    def __len__(self):
        return len(self.image_features)
  
  
    def __getitem__(self, idx):
    
        feature, path, question, answer = self.image_features[idx]
        original_question = question
    
        if isinstance(feature, list):
            X1 = random.choice(feature)   # Data Augmentation for training
        else:
            X1 = feature                  # No Data Augmentation for validation

        X1 = torch.Tensor(X1)
        
        question = preprocess_text(question)
        answer = preprocess_text(answer)[0]
        answer = change(answer)

        padding = ['<pad>']*(self.config['input_seq_len']-len(question))
        question = question + padding
        assert len(question) == self.config['input_seq_len'] , "Length of question is %d"%len(question) 
    

        X2 = np.zeros((self.config['input_seq_len'], self.config['embedding_size']))

        for i in range(self.config['input_seq_len']):
            if question[i] not in self.input_embedding.keys():
                question[i] = '<unk>'
            X2[i] = self.input_embedding[question[i]]

    
        X2 = torch.from_numpy(X2).float()
    
        if answer not in self.word2idx.keys(): answer = '<unk>'
        
        if answer not in self.word_list: answer = '<unk>' 
        
        Y = self.word2idx[answer]

        return X1, X2, Y
