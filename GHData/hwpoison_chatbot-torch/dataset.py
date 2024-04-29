import pandas as pd
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

from torch.utils.data import Dataset

# Se lee el archivo .csv
data = pd.read_csv('database.csv')

all_words = []
tags = []
xy = []

# Se tokenizan las oraciones y se obtienen las palabras
for tag, sentence in data[["tag","sentence"]].values:
    tokenized = tokenize(sentence)
    all_words.extend(tokenized)
    tag = tag.strip()
    xy.append((tokenized, tag)) 
    tags.append(tag)

# Se filtra la información
ignore_words = ["?", "!", ".", ","]
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Se generan los datos de entrenamiento
X_train = []
y_train = []

for pattern, tag in xy:
    bag = bag_of_words(pattern, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

# Se crea el modelo de datos
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
        self.all_words = all_words
        self.tags = tags

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples



