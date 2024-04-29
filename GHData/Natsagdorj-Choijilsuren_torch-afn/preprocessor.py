import argparse
import pandas as pd
import os

import numpy as np


class Processor:

    def __init__(self, data_path):

        self.input_data = pd.read_csv(data_path)
        self.out_data = pd.DataFrame()

        self.field_dict = {}
        self.feature_dict = {}
        
        #CATEGORICAL AND NUMERICAL VALUES

        #TO DO ADD important features 
        self.numerical_cols = ['cnt_date', 'geo_tokyo', 'geo_osaka', 'width', 'height',
                               'age_range_undetermined', 'age_range_18_24', 'age_range_25_34', 'age_range_35_44',
                               'age_range_45_54', 'age_range_55_64', 'age_range_65_more', 'gender_male', 'gender_female',
                               'list_type_rule_based', 'list_type_logical', 'list_type_remarketing', 'list_type_similar',
                               'list_type_crm_based']
        
        self.categorical_cols = ['ad_type', 'status', 'device']

        #binsizes to numerical variable decided by me
        self.numerical_bins = {'cnt_date': 50, 'geo_tokyo':50, 'geo_osaka':50, 'width':10, 'height':10,
                               'age_range_undetermined':20, 'age_range_18_24':20, 'age_range_25_34':20,
                               'age_range_35_44':20, 'age_range_45_54':20, 'age_range_55_64':20, 'age_range_65_more':20,
                               'gender_male':50, 'gender_female':50, 'list_type_rule_based':20, 'list_type_logical':20,
                               'list_type_remarketing':10, 'list_type_similar':10, 'list_type_crm_based':5}

        
    def numerical2categorical(self, col_name, bin_size):
        
        elements = self.input_data[col_name]
        bins, labels = Processor.get_bins(elements, bin_size)
        #TO DO -- O should be in different class itself
        bins[0] = -1
        try:
            self.out_data[col_name] = pd.cut(self.input_data[col_name], bins, labels=labels)
        except:
            self.out_data[col_name] = pd.cut(self.input_data[col_name], bins, labels=labels[:-1])
            
    def categorical2categorical(self, col_name):

        dict_feature = {}
        list_cats = self.input_data[col_name].unique()

        for i, element in enumerate(list_cats):
            dict_feature[element] = i

        def get_feature(element):
            return dict_feature[element]
                
        self.out_data[col_name] = self.input_data[col_name].apply(get_feature)

        
    @staticmethod
    #TO DO make 0 another class itself
    def get_bins(np_arr, bins_size):

        hist, bins = np.histogram(np_arr, bins=bins_size)
        bins = bins.astype(np.int)
        labels = np.arange(bins_size)

        return bins, labels
    
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


    
    
    def get_label(self):

        self.out_data['ctr'] = self.input_data['ctr'].apply(Processor.get_label_preprocess)
        self.out_data['flag'] = self.input_data['flag']
        
    def make_out_dafaframe(self):

        for col_name in self.categorical_cols:
            self.categorical2categorical(col_name)

        for col_name in self.numerical_cols:
            self.numerical2categorical(col_name, self.numerical_bins[col_name])

    def getIndex(self):
        
        self.out_data["idx"] = self.input_data["id"]

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser('Data')

    parser.add_argument('--input_csv', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--mode', type=str, default='train')
    
    args = parser.parse_args()

    process = Processor(args.input_csv)
    
    process.make_out_dafaframe()
    if args.mode == 'train':
        process.get_label()

    else:
        process.getIndex()
        
    process.out_data.to_csv(args.save_path)
    
    
