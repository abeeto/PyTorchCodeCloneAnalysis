import argparse
import pandas as pd
import os

import numpy as np
import sys


class Processor_full:


    def __init__(self, train_data_path, test_data_path):

        self.train_data = pd.read_csv(train_data_path)
        self.test_data = pd.read_csv(test_data_path)
        
        self.out_train = pd.DataFrame()
        self.out_test = pd.DataFrame()
        
        #TO DO ADD important features 
        self.numerical_cols = ['cnt_date', 'geo_tokyo', 'geo_osaka', 'width', 'height',
                               'age_range_undetermined', 'age_range_18_24', 'age_range_25_34', 'age_range_35_44',
                               'age_range_45_54', 'age_range_55_64', 'age_range_65_more', 'gender_male', 'gender_female',
                               'list_type_rule_based', 'list_type_logical', 'list_type_remarketing', 'list_type_similar',
                               'list_type_crm_based']
        
        self.categorical_cols = ['ad_type', 'status', 'device']

        #binsizes to numerical variable decided by me
        #I realized one bug --> train and test bins might differ in this set
        self.numerical_bins = {'cnt_date': 50, 'geo_tokyo':50, 'geo_osaka':50, 'width':10, 'height':10,
                               'age_range_undetermined':20, 'age_range_18_24':20, 'age_range_25_34':20,
                               'age_range_35_44':20, 'age_range_45_54':20, 'age_range_55_64':20, 'age_range_65_more':20,
                               'gender_male':50, 'gender_female':50, 'list_type_rule_based':20, 'list_type_logical':20,
                               'list_type_remarketing':10, 'list_type_similar':10, 'list_type_crm_based':5}
        
    def get_bins(self, col_name, num_bins):

        np_arr_train = self.train_data[col_name].values
        np_arr_test = self.test_data[col_name].values

        np_arr = np.concatenate((np_arr_train, np_arr_test), axis=0)
        hist, bins = np.histogram(np_arr, bins=num_bins)

        labels = np.arange(num_bins + 1)

        return bins, labels
    
    
    def categorical2categorical(self, col_name):
            
        dict_feature = {}
        list_cats = self.train_data[col_name].unique()
                
        for i, element in enumerate(list_cats):
            dict_feature[element] = i

        def get_feature(element):
            return dict_feature[element]

        self.out_train[col_name] = self.train_data[col_name].apply(get_feature)
        self.out_test[col_name] = self.test_data[col_name].apply(get_feature)

        
    @staticmethod
    def get_label_preprocess(ctr_percent):
        #I think this method is good enough
        #sigmoid function is (0, 1)
        out = ctr_percent + 0.0005
        out = np.log(out)

        min_num = -7.600902459542082
        max_num = 4.605175185975591

        out_scaled = (out - min_num)/(max_num - min_num)
        return out_scaled

    def numerical2categorical(self, col_name, num_bins):

        bins, labels = self.get_bins(col_name, num_bins)
        bins = np.insert(bins, 0, -1)

        self.out_train[col_name] = pd.cut(self.train_data[col_name],
                                          bins, labels=labels)
        self.out_test[col_name] = pd.cut(self.test_data[col_name],
                               bins, labels=labels)
        

    def get_label(self):

        self.out_train['ctr'] = self.train_data['ctr'].apply(Processor_full.get_label_preprocess)
        self.out_train['flag'] = self.train_data['flag']

    def getIndex(self):

        self.out_test['idx'] = self.test_data['id']    

    def make_out_dafaframe(self):
        
        for col_name in self.categorical_cols:
            self.categorical2categorical(col_name)
            
        for col_name in self.numerical_cols:
            self.numerical2categorical(col_name, self.numerical_bins[col_name])
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data', type=str, default='../csv_data/20201126_gaw_banner_train.csv')
    parser.add_argument('--test_data', type=str, default='../csv_data/20201126_gaw_banner_test.csv')
    parser.add_argument('--save_train', type=str, default='./train.csv')
    parser.add_argument('--save_test', type=str, default='./test.csv')
    
    args = parser.parse_args()
    
    process = Processor_full(args.train_data, args.test_data)
    process.make_out_dafaframe()
    process.getIndex()
    process.get_label()
    
    process.out_train.to_csv(args.save_train)
    process.out_test.to_csv(args.save_test)
    

    

            
