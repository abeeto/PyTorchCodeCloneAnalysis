import argparse
import random

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import torchvision

import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.quantization import QuantStub, DeQuantStub
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from torchmetrics.functional import accuracy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import TQDMProgressBar
from sklearn.preprocessing import Normalizer

import multiprocessing
#import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
from matplotlib.ticker import MaxNLocator
import random






from data import load_data_set
from data import SequenceDataModule
from data import data_labeler
from data import TimeSeriesDataset
from data import index_scaler

from essentials import other_functions
from essentials import check_path

from model import models
from model import Disturbance_Predictor_model_lstm
from model import LSTM1


## Globals #######
plt.switch_backend('agg')
random.seed(102)

## X train path
x_train_data_path = r"P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\07_df_x_12_500smps_nowind.csv"
y_train_data_path = r"P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\07_df_y_12_500smps_nowind.csv"


#-----------------------------------------------------------------------
## Copy this section for Prediction

# path to safe model
model_architecture = "11_LSTM"
save_model_path = "P:/workspace/jan/fire_detection/dl/models_store/" + model_architecture
check_path(save_model_path)

## model params
N_EPOCHS = 50
BATCH_SIZE = 100
n_features = 8
batch_norm_trigger = False
shuffle = False
save = True
load_model = False

num_layers = 2
hidden_state_size = 300
dropout_rate = 0.5
learning_rate = 0.001

version_name =f'nlayers {num_layers} dropout {dropout_rate} learning_rate {learning_rate} batchsize {BATCH_SIZE} n_epochs {N_EPOCHS} batchnorm {batch_norm_trigger} hiddenstatesize {hidden_state_size} 1fc shuffle {shuffle}'
model_name = f'\\LSTM_{N_EPOCHS}_epochs_{BATCH_SIZE}_batchsize_5classes.pt'

## Copy this section for Prediction
#-----------------------------------------------------------------------



## training logger path
logger_path = save_model_path + "/logger"
check_path(logger_path)
experiment_name = "Disturbances_lstm"


## ---------------------------------------------------------------------------------------------------------------------
## Main

if __name__ ==  '__main__':

    ## read data
    X_train = pd.read_csv(x_train_data_path, sep=';')
    y_train = pd.read_csv(y_train_data_path, sep=';')
    print(X_train.shape, y_train.shape)

    # X_train_2 = index_scaler(X_train)

    # data label
    y_train   = data_labeler(y_train)
    y_train["disturbance"].unique()

    ## data set
    dset = TimeSeriesDataset(X_train, y_train, 25)

    ## split train test
    train_sequences, test_sequences = train_test_split(dset, test_size=0.2, random_state= 102)
    print("Number of Training Sequences: ", len(train_sequences))
    print("Number of Testing Sequences: ", len(test_sequences))

    ## dataloader
    ## here augmentation and normalization could be done
    Data_module = SequenceDataModule(train_sequences, test_sequences, BATCH_SIZE, shuffle)

    ## init logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=logger_path,
                                             name=experiment_name,
                                             version=version_name,
                                             default_hp_metric= False)

    # Model learner
    print("initializing model")
    model = Disturbance_Predictor_model_lstm(
        input_size=n_features ,
        n_classes=len(y_train["label"].unique()),
        num_layers=num_layers,
        learning_rate=learning_rate,
        batch_norm_trigger = batch_norm_trigger,
        batch_size = BATCH_SIZE,
        size_of_hidden_state = hidden_state_size)

    if load_model:
        model = torch.load(save_model_path+model_name)
        print("loaded model")

    trainer = pl.Trainer(
        logger=tb_logger,
        max_epochs=N_EPOCHS,
        callbacks=[TQDMProgressBar(refresh_rate=10)]
    )

    trainer.fit(model, Data_module)

    if save:
        torch.save(model, save_model_path+model_name)
        print("saved model, TSCHAUTSCHAU")


