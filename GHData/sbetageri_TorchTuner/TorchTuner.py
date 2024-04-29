import os
import json
import torch
import pprint

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm


class TorchTuner:
    ## Names for parameters
    _OPTIM_FUNC = 'optimizer_func'
    _OPTIM_PARAM = 'optimizer_param'
    _EPOCHS = 'epochs'
    _BATCH_SIZE = 'batch_size'
    _PARAM_IDX = 'param_id'
    _RESULTS = 'results'
    _TR_LOSS = 'train_loss'
    _TR_ACC = 'train_acc'
    _VAL_LOSS = 'val_loss'
    _VAL_ACC = 'val_acc'
    _MODEL_PATH = 'model_path'

    _BEST_LOSS = 'best_loss'
    _BEST_ACC = 'best_acc'

    _MODE = 'mode'
    _MAX_MODE = 'max'
    _MIN_MODE = 'min'

    CUR_EPOCH =  'current_epoch'
    TOT_EPOCH = 'total_epoch'
    OPTIMIZER_STATE = 'optimizer_state_dict'
    MODEL_STATE = 'model_state_dict'
    MODEL_NAME = 'model_name'
    BATCH_SIZE = 'batch_size'

    MODEL_PREFIX = 'model_'
    EXT = '.pth'
    PARAM_EXT = '.json'
    MODEL_NAME_SEP = '_'


    def __init__(self, 
                 model=None, 
                 model_name='myModel',
                 criterion_func=None,
                 accuracy_func=None,
                 train_dataset=None,
                 test_dataset=None,
                 val_percentage=0.15,
                 res_dir=None):
        '''Initialise torch tuner
        
        :param model: Model to be tested, defaults to None
        :param model: Custom PyTorch model, optional
        :param criterion_func: Criterion Function, defaults to None
        :param criterion_func: torch.nn, optional
        :param accuracy_func: Accuracy funciton, defaults to None
        :param accuracy_func: Custom function, optional
        :param train_dataset: Training dataset, defaults to None
        :param train_dataset: torch.utils.data.Dataset, optional
        :param test_dataset: Test dataset, defaults to None
        :param test_dataset: torch.utils.data.Dataset, optional
        :param val_percentage: Percentage of training set to be used as test set, defaults to 0.15, ie, 15%
        :param val_percentage: float, optional
        :param res_dir: Directory where models and results are saved, defaults to None
        :param res_dir: string, optional
        '''

        self.name = model_name
        self.params = []
        self.param_name_prefix = 'param_'
        self.model = model
        self.criterion = criterion_func()
        self.accuracy_func = accuracy_func
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.results = {}
        self.res_dir = res_dir

        if train_dataset is not None:
            self.train_sampler, self.val_sampler = self._getTrainValSampler(train_dataset, val_percentage)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ## TODO
        ## Test loader will be interesting
        # self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=True)

        # Reporting apparatus
        self.pp = pprint.PrettyPrinter(indent=4)

    def _getTrainValSampler(self, dataset, val_percentage):
        indices = list(range(len(dataset)))
        split = int(val_percentage * len(dataset))
        np.random.seed(9)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
        return train_sampler, val_sampler


    def evaluateModel(self,
                      param_id = None,
                      optimizer_func=None, 
                      optimizer_param=None, 
                      epochs=9,
                      batch_size=4,
                      mode=None):
        '''Evaluates a model and returns loss and accuracy
        
        Builds and executes the entire training and validation pipeline.
        Runs implicitly on the GPU
        
        :param optimizer_func: function to obtain optimizer, defaults to None
        :param optimizer_func: torch.optim, optional
        :param optimizer_param: parameters for optimizer, defaults to None
        :param optimizer_param: dict, optional
        :param epochs: number of epochs, defaults to 9
        :param epochs: int, optional
        :param batch_size: size of batch, defaults to 4
        :param batch_size: int, optional
        :return: Log of evaluation metrics
        :rtype: Dictionary
        '''

        train_loader = DataLoader(dataset=self.train_dataset, 
            batch_size=batch_size, 
            sampler=self.train_sampler)

        val_loader = DataLoader(dataset=self.train_dataset, 
            batch_size=batch_size, 
            sampler=self.val_sampler)


        # Training metrics
        tr_loss = []
        tr_acc = []

        # Validation metrics
        val_loss = []
        val_acc = []

        best_acc = 0.0
        best_loss = 0.0

        model_path = ''

        # Move to GPU
        model = deepcopy(self.model)
        model = model.to(self.device)

        optimizer = optimizer_func(model.parameters(), **optimizer_param)
        criterion = self.criterion

        for e in range(epochs):
            running_acc = 0
            running_loss = 0.0

            model.train(True)

            for data, label in train_loader:
                data = data.to(self.device)
                label = label.to(self.device)

                optimizer.zero_grad()

                output = model(data)
                output = output.view(output.size()[0], -1)

                label = label.view(label.size()[0], -1)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_acc += self.accuracy_func(output, label)

            running_loss /= (len(train_loader) * batch_size)
            running_acc /= (len(train_loader) * batch_size)
            running_acc *= 100

            tr_loss.append(running_loss)
            tr_acc.append(running_acc)

            running_loss = 0.0
            running_acc= 0

            model.eval()
            with torch.no_grad():
                for data, label in val_loader:
                    data = data.to(self.device)
                    label = label.to(self.device)

                    output = model(data)
                    output = output.view(output.size()[0], -1)
                    label = label.view(label.size()[0], -1)

                    loss = criterion(output, label)

                    running_loss += loss.item()
                    running_acc += self.accuracy_func(output, label)

                running_loss /= (len(val_loader) * batch_size)
                running_acc /= (len(val_loader) * batch_size)
                running_acc *= 100

                val_loss.append(running_loss)
                val_acc.append(running_acc)

            if mode == TorchTuner._MAX_MODE:
                if len(val_acc) == 1:
                    model_path = self.saveModel(model, param_id, optimizer, e, epochs, batch_size)
                    best_acc = val_acc[-1]
                    best_loss = val_loss[-1]
                elif val_acc[-1] > val_acc[-2]:
                    best_acc = val_acc[-1]
                    best_loss = val_loss[-1]
                    print('Accuracy improved. Saving model')
                    model_path = self.saveModel(model, param_id, optimizer, e, epochs, batch_size)
            else:
                if len(val_loss) == 1:
                    best_acc = val_acc[-1]
                    best_loss = val_loss[-1]
                    model_path = self.saveModel(model, param_id, optimizer, e, epochs, batch_size)
                elif val_loss[-1] < val_loss[-2]:
                    best_acc = val_acc[-1]
                    best_loss = val_loss[-1]
                    print('Loss decreased. Saving model')
                    model_path = self.saveModel(model, param_id, optimizer, e, epochs, batch_size)

        return {
            TorchTuner._TR_LOSS : tr_loss,
            TorchTuner._TR_ACC : tr_acc,
            TorchTuner._VAL_LOSS : val_loss,
            TorchTuner._VAL_ACC : val_acc,
            TorchTuner._BEST_ACC : best_acc,
            TorchTuner._BEST_LOSS : best_loss,
            TorchTuner._MODEL_PATH : model_path
        }

    def addModel(self, 
                model=None,
                criterion_func=None,
                accuracy_func=None,
                clear_prev=False):
        '''Change model
        
        Change the underlying model for which the hyperparameters need to be tested
        
        :param model: Pytorch model, defaults to None
        :param model: Custom model, optional
        :param criterion_func: Loss function, defaults to None
        :param criterion_func: torch.nn, optional
        :param accuracy_func: Evaluation metric function, defaults to None
        :param accuracy_func: function, optional
        '''

        self.model = model
        self.criterion_func = criterion_func
        self.accuracy_func = accuracy_func
        if clear_prev:
            self.results = {}

    def addHyperparameters(self,
                           optimizer_func=None,
                           optimizer_param=None,
                           epochs=9,
                           batch_size=8,
                           mode=None):
        '''Add hyperparams for evaluation
        
        :param optimizer_func: Optimizer, defaults to None
        :param optimizer_func: torch.optim, optional
        :param optimizer_param: Parameters to optimizer, defaults to None
        :param optimizer_param: Dict of params, optional
        :param epochs: Number of epochs to run evaluation metric on, defaults to 9
        :param epochs: int, optional
        :param batch_size: Number of data-points to consider during evaluation, defaults to 8
        :param batch_size: int, optional
        :param mode: Defines which metric to use when saving model, defaults to 'max', max accuracy
        :param mode: str, optional
        '''

        param_idx = self.param_name_prefix + str(len(self.params) + 1)
        param = {
            TorchTuner._PARAM_IDX : param_idx,
            TorchTuner._OPTIM_FUNC : optimizer_func,
            TorchTuner._OPTIM_PARAM : optimizer_param,
            TorchTuner._EPOCHS : epochs,
            TorchTuner._MODE : mode
        }
        self.params.append(param)

    ## How should the parameters be?
    def evaluateHyperparams(self):
        '''Evaluate hyperparameters
        
        Evaluate hyperparams and log results
        
        '''

        self.results = deepcopy(self.params)
        for param in self.results:
            result = self.evaluateModel(**param)
            param[TorchTuner._RESULTS] = result
    
    def saveHyperparam(self,
                       out_file=''):
        '''Save hyperparameters to json file
        
        :param out_file: Path to output file, defaults to './param.json'
        :param out_file: str, optional
        '''
        
        # Change results to savable format
        for param in self.results:
            param[TorchTuner._OPTIM_FUNC] = str(param[TorchTuner._OPTIM_FUNC])

        out_file = out_file + TorchTuner.PARAM_EXT
        
        with open(out_file, 'w') as fp:
            json.dump(self.results, indent=4, sort_keys=True, fp=fp)
    
    def saveModel(self,
                  model,
                  param_id,
                  optimizer,
                  cur_epoch,
                  total_epoch,
                  batch_size):

        ## We're passing model again because model here is supposed to the model on the GPU.
        save_dict = {
            TorchTuner.MODEL_STATE : model.state_dict(),
            TorchTuner.OPTIMIZER_STATE : optimizer.state_dict(),
            TorchTuner.CUR_EPOCH : cur_epoch,
            TorchTuner.TOT_EPOCH : total_epoch,
            TorchTuner.BATCH_SIZE : batch_size,
        }
        model_name = self.name + TorchTuner.MODEL_NAME_SEP + param_id + TorchTuner.EXT
        res_dir = os.path.abspath(self.res_dir)
        model_path = os.path.join(res_dir, model_name)
        torch.save(save_dict, model_path)
        return model_path

    def testModel(self, param):
        result = param[TorchTuner._RESULTS]
        model_path = result[TorchTuner._MODEL_PATH]
        checkpoint = torch.load(model_path)
        model_state_dict = checkpoint[TorchTuner.MODEL_STATE]
        model = deepcopy(self.model)
        model = model.to(self.device)
        model.load_state_dict(model_state_dict)

        batch_size = checkpoint[TorchTuner.BATCH_SIZE]

        test_loader = DataLoader(dataset=self.test_dataset, 
            batch_size=batch_size, 
            shuffle=True)

        running_acc = 0.0
        running_loss = 0.0

        for data, label in test_loader:
            data = data.to(self.device)
            label = label.to(self.device)

            output = model(data)
            output = output.view(output.size()[0], -1)
            label = label.view(label.size()[0], -1)

            loss = self.criterion(output, label)

            running_loss += loss.item()
            running_acc += self.accuracy_func(output, label)

        running_loss /= (len(test_loader) * batch_size)
        running_acc /= (len(test_loader) * batch_size)
        running_acc *= 100

        return running_loss, running_acc