#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# License: © 2022 Achille-Tâm GUILCHARD All Rights Reserved
# Author: Achille-Tâm GUILCHARD

import argparse
from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
import optuna
from optuna import Trial

import multiprocessing
from termcolor import colored


def parse_arguments():
    """Parse input args"""                                                                                                                                                                                                                            
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--n_trials', type=int, default=30, help='Number of trials to do.', required=True)
    return parser.parse_args() 


def load_data(data_dir="./dataset"):
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(229),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'normal': transforms.Compose([
            transforms.Resize(229),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets       = datasets.ImageFolder(data_dir, data_transforms['normal'])
    train_size           = int(0.85 * len(image_datasets))
    
    test_size            = len(image_datasets) - train_size
    
    trainset, testset    = torch.utils.data.random_split(image_datasets, [train_size , test_size])

    return trainset, testset

def train(param, trial, target, nb_classes=3, multi_objective_optimization=False, checkpoint_dir=None, data_dir=None):
    net         = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.DEFAULT, progress=True)
    num_ftrs    = net.fc.in_features
    net.fc      = nn.Linear(num_ftrs, nb_classes)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"])
    optimizer = None
    if param['optimizer'] == "SGD":
        optimizer = getattr(optim, param['optimizer'])(net.parameters(), lr=param['learning_rate'], momentum=param['momentum'])
    else:
        optimizer = getattr(optim, param['optimizer'])(net.parameters(), lr=param['learning_rate'])

    trainset, testset        = load_data(data_dir)

    test_abs                 = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(param["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(param["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(param["epoch_num"]):  # loop over the dataset multiple times
        print("\nEPOCH {} of {}".format(epoch+1, param["epoch_num"]))
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d,%5d] loss: %.9f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss  = 0.0
        val_steps = 0
        total     = 0
        correct   = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs      = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total       += labels.size(0)
                correct     += (predicted == labels).sum().item()

                loss       = criterion(outputs, labels)
                val_loss  += loss.cpu().numpy()
                val_steps += 1

        loss     = (val_loss / val_steps)
        accuracy = correct / total
        print(f" > Epoch #{epoch+1} validation accuracy: {accuracy:.9f}, validation loss: {loss:.9f}")

        if multi_objective_optimization == False:
            if target["to_return"] == "loss":
                trial.report(loss, epoch+1)
            elif target["to_return"] == "accuracy":
                trial.report(accuracy, epoch+1)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return accuracy, loss


def test_accuracy(net, device="cpu", data_dir="./dataset"):
    trainset, testset = load_data(data_dir)
    testloader        = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    correct = 0
    total   = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


 # Define a set of hyperparameter values, build the model, train the model, and evaluate the accuracy 
def objective(trial):
    params = {
              'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
              'optimizer':     trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
              'momentum':      trial.suggest_uniform('momentum', 0.1, 0.9),
              'batch_size':    trial.suggest_categorical('batch_size', [2, 4, 8, 16]),
              'epoch_num':     trial.suggest_categorical('epoch_num', [1, 2, 3, 4, 5])
              }
    
    nb_classes     = len(next(os.walk('./dataset'))[1])
    multi_objective_optimization = False
    target = {"to_return": "loss"}
    accuracy, loss = train(params, trial, target, nb_classes=nb_classes, multi_objective_optimization=multi_objective_optimization, data_dir='./dataset')

    if multi_objective_optimization:
        return loss, accuracy 
    else:
        if target["to_return"] == "loss":
            return loss
        else:
            return accuracy


def main(n_trials=30):
    
    # study = optuna.create_study(directions="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.SuccessiveHalvingPruner())
    # directions = ["minimize", "maximize"]
    directions = ["minimize"]
    
    study = optuna.create_study(directions=directions, sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
    
    study.optimize(objective, n_trials)

    if len(directions) == 1:
        best_trial = study.best_trial
        print("Best hyperparams:")
        for key, value in best_trial.params.items():
            print("{}: {}".format(key, value))

        imp = optuna.importance.get_param_importances(study)
        print(imp)

    else:
        best_trials = study.best_trials
        # print(best_trials)

        imp = optuna.importance.get_param_importances(study, target=lambda t: t.values[0])
        print(imp)

        # fig = optuna.visualization.plot_pareto_front(study)
        # fig.write_image("./pareto_front.png")

if __name__ == "__main__":
    args = parse_arguments()
    print("Number of trial: {}".format(args.n_trials))
    main(args.n_trials)
