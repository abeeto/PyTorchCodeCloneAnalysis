import const
import dataParser
from functions import *
from features.grams import *
import configuration as config
import numpy as np
from features.extractFeatures import *
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data
import const



#extract the data from csv
const.M_names, const.F_names, const.name_map = dataParser.get_data()
#extract the training/test sets
const.ngram_training_set = extract(config.data_extraction_size)
const.training_set = extract(config.training_size, labelled=True)
const.cv_set = extract(config.cv_set_size, labelled=True)
const.test_set = extract(config.test_set_size, labelled=True)
#determine most common ngrams

getCommonGrams(const.ngram_training_set)
get_suffixes(const.ngram_training_set)

const.featureCount += config.di_num + config.tri_num + config.last_letters + config.di_sufnum + config.tri_sufnum

const.X_train, const.y_train = extractFeatures(const.training_set)
const.X_cv, const.y_cv = extractFeatures(const.cv_set)
const.X_test, const.y_test = extractFeatures(const.test_set)


#--CLASSIFER--#

X_train = torch.stack([torch.tensor(i) for i in const.X_train])
y_train = torch.from_numpy(const.y_train)

X_cv = torch.stack([torch.tensor(i) for i in const.X_cv])
y_cv = torch.from_numpy(const.y_cv)

X_test = torch.stack([torch.tensor(i) for i in const.X_test])
y_test = torch.from_numpy(const.y_test)

training_set = data.TensorDataset(X_train, y_train)
training_loader = data.DataLoader(training_set, batch_size=config.minibatch, shuffle=True)

cv_set = data.TensorDataset(X_cv, y_cv)
cv_loader = data.DataLoader(cv_set, batch_size=config.minibatch, shuffle=False)

test_set = data.TensorDataset(X_test, y_test)
test_loader = data.DataLoader(test_set, batch_size=config.minibatch, shuffle=False)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(const.featureCount, 1)

    def forward(self, X):
        return self.linear(X)

model = Model()
cost = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
epochs = int(config.iterations / config.training_size * config.minibatch)

iterations = 0
for i in range(epochs):
    for names, genders in training_loader:
        names = names.view(-1, const.featureCount).requires_grad_()
        optimizer.zero_grad()
        hypothesis = torch.sigmoid(model(names.float()))
        loss = cost(hypothesis.reshape(config.minibatch), genders.float())
        loss.backward()
        optimizer.step()
        iterations += 1
        if not iterations % 500:
           for num, set in enumerate([cv_loader, test_loader]):
                correct = 0
                total = 0
                for name, gender in set:
                    name = name.view(-1, const.featureCount).requires_grad_()
                    pred = torch.sigmoid(model(name.float()))

                    total += gender.size(0)
                    correct += ((pred.reshape(config.minibatch)-gender.reshape(config.minibatch)).abs_() < 0.5).sum()

                accuracy = 100 * correct.item() / total
                type = "Test" if num else "CV"
                print('Type: {}- Iteration: {}. Cost: {}. Accuracy: {}'.format(type, iterations, loss.item(), accuracy))
            # print("")

if config.IS_DEBUG:
    for name, param in model.named_parameters():
        if param.requires_grad and config.IS_DEBUG:
            print(name)
            weights = param.data
            for name, weight in zip(const.featureList, weights[0]):
                print(f"[{name}] -> {weight}")
            config.IS_DEBUG = False
#now we can test custom name data

name_map_custom = dataParser.get_custom_data()

X_custom, y_custom = extractFeatures(name_map_custom)

X_custom_tensor = torch.stack([torch.tensor(i) for i in X_custom])
y_custom_tensor = torch.from_numpy(y_custom)

custom_set = data.TensorDataset(X_custom_tensor, y_custom_tensor)
custom_loader = data.DataLoader(custom_set, batch_size=config.minibatch)



for name, gender in custom_loader:
    name = name.view(-1, const.featureCount).requires_grad_()
    pred = torch.sigmoid(model(name.float()))

    c_total = gender.size(0)
    predictions = (pred.reshape(len(name_map_custom)) - gender.reshape(len(name_map_custom))).abs_() < 0.5
    for index, name in enumerate(name_map_custom.keys()):
        print('Name: {}. Gender: {}. Prediction: {}'.format(name, "M" if gender[index].item() else "F", "M" if predictions[index].item() else "F"))
