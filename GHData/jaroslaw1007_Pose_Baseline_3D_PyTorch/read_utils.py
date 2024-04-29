import numpy as np
import pandas as pd
import os
import torch
import glob

from cameras import *
from config import *
from utils import *

def read_stats_data(dim=3):
    if dim == 2:
        stats_2d = pd.read_csv(os.path.join(STATS_PATH, 'stats_2d.csv'))
        stats_2d_dim = pd.read_csv(os.path.join(STATS_PATH, 'stats_2d_dim.csv'))
        
        return stats_2d['x_mean'].to_numpy(), stats_2d['x_std'].to_numpy(),\
               stats_2d_dim['x_dim_to_use'].to_numpy(), stats_2d_dim['x_dim_to_ignore'].to_numpy()
    else:
        stats_3d = pd.read_csv(os.path.join(STATS_PATH, 'stats_3d.csv'))
        stats_3d_dim = pd.read_csv(os.path.join(STATS_PATH, 'stats_3d_dim.csv'))
        
        return stats_3d['y_mean'].to_numpy(), stats_3d['y_std'].to_numpy(),\
               stats_3d_dim['y_dim_to_use'].to_numpy(), stats_3d_dim['y_dim_to_ignore'].to_numpy()

def load_data(base_path, subjects, actions, dim=3):
    data = {}

    for subject in subjects:
        for action in actions:
            data_path = os.path.join(base_path, 'S{0}'.format(subject), 'MyPoses/{0}D_positions'.format(dim), '{0}*.h5'.format(action))
            file_names = glob.glob(data_path)

            for file_name in file_names:
                seq_name = os.path.basename(file_name)

                if action == 'Sitting' and seq_name.startswith('SittingDown'):
                    continue
                if seq_name.startswith(action):
                    with h5py.File(file_name, 'r') as h5f:
                        poses = h5f['{0}D_positions'.format(dim)][:]

                    poses = poses.T
                    data[(subject, action, seq_name)] = poses
    return data

def read_3d_data(data_dir, actions, rcams, is_train=True):
    train_dataset = load_data(data_dir, TRAIN_SUBJECTS, actions, dim=3)
    
    train_dataset = transform_world_to_camera(train_dataset, rcams) 
    
    # Centering around root
    train_dataset, train_root_coords = postprocess_3d(train_dataset)
    
    # Calculate data mean, std and dimensions to use, ignore
    # complete_train = copy.deepcopy(np.vstack(train_dataset.values()))
    # data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats(complete_train, dim=3)
    data_mean, data_std, dim_to_use, dim_to_ignore = read_stats_data(dim=3)
    
    train_dataset = normalize_data(train_dataset, data_mean, data_std, dim_to_use)
    
    if is_train:
        return train_dataset
    else:
        test_dataset = load_data(data_dir, TEST_SUBJECTS, actions, dim=3)
        test_dataset = transform_world_to_camera(test_dataset, rcams)
        test_dataset, test_root_coords = postprocess_3d(test_dataset)
        test_dataset = normalize_data(test_dataset, data_mean, data_std, dim_to_use)
        
        return test_dataset

def read_2dgt_data(data_dir, actions, rcams, is_train=True):
    train_dataset = load_data(data_dir, TRAIN_SUBJECTS, actions, dim=3)
    
    train_dataset = project_to_cameras(train_dataset, rcams)
    
    # Calculate data mean, std and dimensions to use, ignore
    # complete_train = copy.deepcopy(np.vstack(train_dataset.values()))
    # data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats(complete_train, dim=2)
    data_mean, data_std, dim_to_use, dim_to_ignore = read_stats_data(dim=2)
    
    train_dataset = normalize_data(train_dataset, data_mean, data_std, dim_to_use)
    
    if is_train:
        return train_dataset
    else:
        test_dataset = load_data(data_dir, TEST_SUBJECTS, actions, dim=3)
        test_dataset = project_to_cameras(test_dataset, rcams)
        test_dataset = normalize_data(test_dataset, data_mean, data_std, dim_to_use)

    return test_dataset