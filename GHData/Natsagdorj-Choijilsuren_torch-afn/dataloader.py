import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset


class TrainDataset(Dataset):

    def __init__(self, pd_data):

        self.pd_data = pd_data

        self.target_column = 'ctr'
        
        self.feature_cols = ['cnt_date', 'geo_tokyo', 'geo_osaka', 'width', 'height',
                               'age_range_undetermined', 'age_range_18_24', 'age_range_25_34', 'age_range_35_44',
                               'age_range_45_54', 'age_range_55_64', 'age_range_65_more', 'gender_male', 'gender_female',
                               'list_type_rule_based', 'list_type_logical', 'list_type_remarketing', 'list_type_similar',
                               'list_type_crm_based', 'ad_type', 'status', 'device']
        
        
    def __getitem__(self, idx):

        feature = self.pd_data[self.feature_cols].iloc[idx].to_list()
        target = self.pd_data[self.target_column].iloc[idx]

        return np.array(feature), target
    
    def __len__(self):

        return len(self.pd_data)

    def get_field_dims(self):

        dims = []

        for col_name in self.feature_cols:
            dims.append(len(self.pd_data[col_name].unique()))
            
        print (dims)

        return dims

        
    @classmethod
    def get_from_one_file(cls, data_path, train_flag = '9 10'):
        
        pd_data = pd.read_csv(data_path)
        
        val_flags = [int(number) for number in train_flag.split(' ')]
        train_data = pd_data[pd_data.flag]
        
        
def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument()
    parser.add_argument()

    args = parser.parse_args()

    return args



if __name__ == '__main__':

    pd_data = pd.read_csv('./out.csv')
    dataset = TrainDataset(pd_data)

    dataset.get_field_dims()
    
