import datetime
import os
import sys

import torch

DEBUG = True if sys.gettrace() is not None else False
device = "cuda" if torch.cuda.is_available() else "cpu"

dtm = datetime.datetime.now().strftime("%d-%H-%M-%S-%f")

DATASET_PATH = os.path.join("data", "train.npy")
ORIGINAL_PATH = "data/elespino/elespino_dataset.csv"

LOGS = "logs" if not DEBUG else "tmp"
LOGS = os.path.join(LOGS, dtm)
# RECORD = os.path.join(LOGS, "record")
MODEL = os.path.join(LOGS, "model")
TENSORBOARD = os.path.join(LOGS, "tensorboard")
PLOTS = os.path.join(LOGS, "plots")
# METRIC_LOGGER = os.path.join(LOGS, "metric")

# os.makedirs(RECORD, exist_ok=True)
# os.makedirs(METRIC_LOGGER, exist_ok=True)


def make_paths():
    os.makedirs(MODEL, exist_ok=True)
    os.makedirs(TENSORBOARD, exist_ok=True)
    os.makedirs(PLOTS, exist_ok=True)


class Training:
    max_epochs = 100
    proportion = 0.8
    batch_size = 32

    save_every = 2
    log_every = 10
    eval_every = 2
