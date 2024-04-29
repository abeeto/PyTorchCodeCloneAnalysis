# Seth Michel

import pandas as pd
import os
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from skorch import NeuralNetClassifier


class MyModule(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu):
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X))
        
        return X


def parseFiles():
    global KaggleData_df

    KaggleData = pd.read_csv("datasets/kaggleDataSets/Fake.csv")

    KaggleData_df = KaggleData[['title']].values.tolist()

    L = round(len(KaggleData_df) * 0.7)

    return KaggleData_df[:L]


BATCH_SIZE = 16

if (not os.path.isdir('./.data')):
  os.mkdir('./.data')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

testList = parseFiles()

# Load the spacy model that you have installed
nlp = spacy.load('en_core_web_sm')
nlp.max_length = 2800000    # my stuff is too long, default is 1,000,000

# Get the vector for 'text':
pDoc = nlp(str(testList))

net = NeuralNetClassifier(
    MyModule,
    max_epochs=50,
    lr=0.1,
    iterator_train__shuffle=True,   # Shuffle training data on each epoch
    )

params = {
    'lr': [0.01, 0.02],
    'max_epochs': [5, 50],
    'module__num_units': [10, 20],
    }

# train the model
gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy')

gs.fit(X, y)

# print results
print(gs.best_score_, gs.best_params_)