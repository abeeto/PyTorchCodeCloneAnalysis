#!/usr/bin/env python
# coding: utf-8

import sys
from tensorboardX import SummaryWriter
import torch
import numpy as np
import pandas as pd


def main():
    """takes command line arguement meta-data file name followed 
    by vectors coressponding to the meta-data, note the data should be saved in tab
    seperated format (delimiter= '\t')

    Example uses:
    >>python embedding.py meta_data.tsv vectors.tsv
    >>tensorboard logdir= ./

    Go to the localhost:6060 or (based on  output in terminal) in browser & 
    select projector from drop down menu to visualize the embedding
    """
    
    read_data = sys.argv
    print(read_data)
    # read metadata
    label = pd.read_csv(str(read_data[1]),sep='\t')
    # read vectors
    data = pd.read_csv(str(read_data[2]),sep='\t')

    # converting to numpy array
    label = np.array(label) 
    data = np.array(data)

    # reshape the meta-data
    label= label.reshape(label.shape[0])


    writer = SummaryWriter()
    writer.add_embedding(data,metadata=label)
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    print("Working")

main()