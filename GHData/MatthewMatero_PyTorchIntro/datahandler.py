import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

def open_data(table):
    """
        Load data from Database or CSV file
        Convert into pandas dataframe for later batching
    """
    engine = create_engine(myDB, encoding='latin1') 
    conn = engine.connect()
    select = conn.execute('select * from ' + table)

    df = pd.DataFrame(select.fetchall()) 
    df.columns = select.keys()

    conn.close()
    return df

def rand_data():
    """
        Generates random data in Tensors
    """
    # 100 examples, with seq_len=10, each holding 300 features
    return torch.randn((100, 10, 300))

def get_labels():
    return torch.randint(0,3, (100,1), dtype=torch.int64)

def get_regression_labels():
    # 100 random numbers following a normal dist between 0-1
    return torch.randn((100,1))

def build_dataloader(bs, shfle):
    """
        Builds a PyTorch Dataloader object

        args:
            bs - (integer) number of examples per batch
            shfle - (bool) to randomly sample train instances from dataset
    """
    # change get_labels to correct version (classification vs regression)
    dataset = TensorDataset(rand_data(), get_labels())
    dataset = TensorDataset(rand_data(), get_regression_labels())

    return DataLoader(dataset, batch_size=bs, shuffle=shfle, num_workers=0)
