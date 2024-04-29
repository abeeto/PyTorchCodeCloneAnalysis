import pandas as pd
import numpy as np
from glob import glob
import re
from invisible_cities.io.dst_io import load_dst, load_dsts
def create_labels_data(list_filenames, en_range=(1.4,1.9)):
    full_csv = pd.DataFrame(columns=['filename', 'event', 'evt_energy', 'evt_ntrks', 'evt_r_max', 'evt_z_min', 'evt_z_max'])
    for filename in list_filenames:
        summary_df = load_dst(filename, 'Summary', 'Events')
        if summary_df is  None:
            print('skipping empty {}'.format(filename))
            continue
        summary_df = summary_df[['event', 'evt_energy', 'evt_ntrks', 'evt_r_max', 'evt_z_min', 'evt_z_max']][(summary_df.evt_energy>=en_range[0]) & (summary_df.evt_energy<=en_range[1])]
        summary_df.loc[:, 'filename'] = filename
        full_csv = pd.concat([full_csv, summary_df], ignore_index=True)
    #full_csv.to_csv(csv_files_name, index=False)                                                                                                                
    return full_csv

def create_dataset_data():
    files_data = glob('/lustre/ific.uv.es/ml/ific020/data_runs/*/cdst/*.h5')
    full_csv = create_labels_data(files_data)
    full_csv.loc[:,'run_number'] = full_csv['filename'].apply(lambda x : int((re.findall("\d+",x )[1])))
    full_csv.loc[:, 'label'] = 0
    train_df_csv = full_csv[full_csv.run_number!=7470]
    valid_df_csv = full_csv[full_csv.run_number==7470]
    print(train_df_csv.shape, valid_df_csv.shape)
    return train_df_csv, valid_df_csv

def create_dataset_MC():
    labels = pd.read_csv('//lustre/ific.uv.es/ml/ific020/MC/labels_MC.csv')
    labels.loc[:,'run_number'] = labels['filename'].apply(lambda x : int((re.findall("\d+",x )[-2])))
    labels.loc[:, 'label'] = 1* ((labels.positron==1) & (labels.E_add==0))
    msk_train = (labels.evt_energy>1.35) & (labels.evt_energy<2.1)
    labels = labels[msk_train]
    train_df = labels[labels.run_number!=6206].copy().reset_index(drop=True)
    valid_df = labels[labels.run_number==6206].copy().reset_index(drop=True)
    return train_df, valid_df

def create_domain_dataset(data_train, MC_train, data_valid, MC_valid):
    MC_domain = MC_train[list(data_train.columns)].copy()
    MC_domain.label = 1
    MC_domain_valid = MC_valid[list(data_train.columns)].copy()
    MC_domain_valid.label = 1
    domain_df = pd.concat([data_train, MC_domain], ignore_index=True)
    valid_domain_df = pd.concat([data_valid, MC_domain_valid], ignore_index=True)
    return domain_df, valid_domain_df


def main():
    hdf_file_labs = '/lustre/ific.uv.es/ml/ific020/dataset_labels.h5'
    data_train, data_valid = create_dataset_data()
    MC_train, MC_valid = create_dataset_MC()
    
    data_train.to_hdf(hdf_file_labs, key='data_train', mode='w')
    data_valid.to_hdf(hdf_file_labs, key='data_valid')
    MC_train   .to_hdf(hdf_file_labs, key='MC_train')
    MC_valid   .to_hdf(hdf_file_labs, key='MC_valid')

    domain_train, domain_valid = create_domain_dataset(data_train, MC_train, 
                                                       data_valid, MC_valid)
    
    domain_train.to_hdf(hdf_file_labs , key='domain_train')
    domain_valid.to_hdf(hdf_file_labs , key='domain_valid')
    
if __name__ == "__main__":
    main()
