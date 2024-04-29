""" Dataset Management Module """

import logging
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from app.helpers.training import train
from app.helpers.dataset import load_pandas_dataset, create_tensors
from app.network.classifier import Classifier

logging.basicConfig(level=logging.INFO)

dataset = load_pandas_dataset()
X, y = create_tensors(dataset)

nb_classes = 2
input_size = len(X[0])

model = Classifier(input_size, nb_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train(model, X, y, criterion, optimizer)

accuracy = accuracy_score(model.predict(X), y)
logging.info("Accuracy is: %s", accuracy)
