import argparse
import random

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.quantization import QuantStub, DeQuantStub
from torch.utils.data import Dataset, DataLoader
import torchvision
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
import seaborn as sns
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

import matplotlib.dates as mdates
import os
from datetime import datetime




## Globals #######
plt.switch_backend('agg')
random.seed(102)

## set paths
ref_path = r'P:\workspace\jan\fire_detection\disturbance_ref\bb_timesync_reference_with_wind.csv'
ts_data_path = r'P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\04_df_x_800000_ts_long_2.csv'

## set globals
window_len=25
model_name = 'LSTM'
refs_to_plot = ['Harvest','Fire','Insect','Wind']


#-----------------------------------------------------------------------
## Copy this section
# path to safe model

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

## Copy this section
#-----------------------------------------------------------------------

wind = False

## out paths
base_path = f'P:/workspace/jan/fire_detection/dl/plots/{model_architecture}/'
check_path(base_path)
name_base = version_name + '.jpg'


trained_model = torch.load(save_model_path + model_name)
trained_model.freeze()


## load references
with open(ref_path, 'r') as f:
    refs = pd.read_csv(f, delimiter=';')

refs = refs[refs['change_process'].isin(refs_to_plot)]
refs['change_process'].unique()

## load time series
with open(ts_data_path, 'r') as f:
    X_ts = pd.read_csv(f, delimiter=';',
                       dtype={
                           'x' :   'int64',
                           'y' :    'int64',
                           'id':    'int64',
                           'date' :  'object',
                           'sensor':  'object',
                           'value':    'float32',
                           'valuenorm': 'float32',
                           'tile':    'object',
                            'index' :  'object',
                            'change_process': 'object',
                             'diff' :  'float64',
                              'instance': 'float64',
                            'time_sequence' : 'int64'
                       }
                       )

## get unique ids from ts data
ids = X_ts['id'].unique()



plt.switch_backend('agg')
## loop over ids
for id in ids[20:40]:
    print(id)
    #id = 1059

    ## subset df for each id
    sample = X_ts[X_ts['id']==id]

    ## store tile id of sample
    tile_id = sample['tile'].values[0]

    ## into long format
    sample = sample.pivot(index='index', columns='date', values='value')

    index = list(sample.index)
    dates = list(sample.columns)
    sample = sample.dropna(axis = 1, how = 'any').astype('float32')

    ## into np
    sample_np = sample.to_numpy(dtype = 'float32')

    #sample_np.dtype
    sample_windowed = np.lib.stride_tricks.sliding_window_view(sample_np, (len(index),window_len ))
    sample_windowed = sample_windowed[0,:,:,:]
    ## predict for windows
    sample_windowed

    predictions = []
    n_windows = sample_windowed.shape[0]
    for i_window in range(n_windows): ## ttodo into list comprehension
        print(i_window)

        sample_windowed_tensor = torch.from_numpy(sample_windowed[i_window,:,:])


        _, output = trained_model(sample_windowed_tensor.unsqueeze(dim=0))
        prediction = torch.softmax(output, dim=1)
        prediction_np = prediction.numpy()
        if i_window == 0:
            print("here")
            predictions = prediction_np
        else :

            #predictions = np.concatenate((predictions,prediction_np[0,:]), axis=0)
            predictions = np.vstack((predictions, prediction_np[0, :]))


    # plot it
    ## time series
    file_name = str(id) + name_base
    ## init plot
    #figure(figsize=(30, 5), dpi=100)
    fig, (ax_1, ax_2) = plt.subplots(2, figsize = (30,10),sharex=True)
    fig.suptitle(model_name + ' prediction and ts of plot id: ' + str(id), fontsize = 18)

    ## set dates
    formatter = mdates.DateFormatter("%Y")  ### formatter of the date
    locator   = mdates.YearLocator()  ### where to put the labels

    ax_2.xaxis.set_major_formatter(formatter)  ## calling the formatter for the x-axis
    ax_2.xaxis.set_major_locator(locator)  ## calling the locator for the x-axis

    #dates =  datetime.strptime(list(sample.columns), "%Y-%m-%d")
    dates =  [datetime.strptime(date, "%Y-%m-%d") for date in list(sample.columns)]
    #ax1.plot(2, 1, 1)
    ax_2.plot(dates, sample.T)
    ax_2.legend(list(sample.index), fontsize = 16,bbox_to_anchor=(1.01, 1), loc="upper left")
    #ax2.tight_layout()

    ## predictions
    ## get dates
    dates_prediction = dates[int((window_len - 1) / 2):(len(dates)-int((window_len-1) / 2))]

    ## get predictions
    #probs_np = probs.numpy()

    ax_1.xaxis.set_major_formatter(formatter)  ## calling the formatter for the x-axis
    ax_1.xaxis.set_major_locator(locator)  ##


    h = ax_1.plot(dates_prediction, predictions)
    h[4].set_alpha(0.4)

    if wind:
        h[5].set_alpha(0.7)
    h[4].set_linestyle('dashdot')

    if wind:
        h[5].set_linestyle(':')

    h[0].set_color("red")
    h[1].set_color("brown")
    h[2].set_color("goldenrod")
    h[3].set_color("blue")
    h[4].set_color("black")

    if wind:
        h[5].set_color("green")

    ax_1.legend(['fire','harvest','insect','wind','stable','growth'], fontsize=16,bbox_to_anchor=(1.01, 1), loc="upper left")

    ## get refs
    ref_id = refs[(refs["plotid"] == id) & (refs["disturbance"]==1) ]
    dates_refs = [datetime.strptime(ref_d, "%Y-%m-%d") for ref_d in list(ref_id["change_date"])]
    change_label = list(ref_id["change_process"])

    ## add refs
    y_min, y_max = ax_1.get_ylim()
    ax_1.vlines(x=dates_refs, ymin=y_min, ymax=y_max, color='k', ls='--')
    y_min, y_max = ax_2.get_ylim()
    ax_2.vlines(x=dates_refs, ymin=y_min, ymax=y_max, color='k', ls='--')

    ## add ref text
    [ax_2.text(dates_refs[i],y= (y_max - (y_max / 12 ) ) , s = change_label[i],
               fontfamily = 'monospace', fontsize = 'xx-large',fontstyle = 'italic',
               verticalalignment = 'top', fontweight = 'roman', ha = 'right')
     for i in range(len(dates_refs))]

    # rotation = 45,
    ax_1.tick_params(axis='both', which='major', labelsize=16)
    ax_1.tick_params(axis='both', which='minor', labelsize=16)
    ax_2.tick_params(axis='both', which='major', labelsize=16, rotation = 0,)
    ax_2.tick_params(axis='both', which='minor', labelsize=16, rotation = 0)
    ax_2.tick_params(axis='x', which='major', labelsize=16, rotation = 45)




    ## set path name
    tile_path = os.path.join(base_path, tile_id)
    if os.path.exists(tile_path):
        print("path exists")
    else:
        os.mkdir(tile_path)

    fig.tight_layout()
    #figure(figsize=(30, 5), dpi=100)
    print("plot..")
    plt.savefig(os.path.join(tile_path, file_name), dpi = 100)
    plt.clf()
