import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nlp_dataset import *
import numpy as np
from codecarbon import EmissionsTracker
from torch import nn
from transformers import BertModel,BertTokenizer
from torchsummary import summary
from pthflops import count_ops
from torch.optim import Adam
from tqdm import tqdm
import random


tracker = EmissionsTracker()


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


ds = TextClassificationDataset(100, 100, 10)
dl = DataLoader(ds, batch_size = 32)
input_size = (1, 128)

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 10)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

model = BertClassifier()


def train(model, learning_rate, epochs):

    train = TextClassificationDataset()

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
                output = model(input_id, mask)
                
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train): .3f} \
                | Train Accuracy: {total_acc_train / len(train): .3f}')
                  
EPOCHS = 5
LR = 1e-6
              
tracker.start()
train(model, LR, EPOCHS)
tracker.stop()
