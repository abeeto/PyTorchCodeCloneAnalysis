import glob, os, sys, pdb, time
import pandas as pd
import numpy as np
import cv2
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from imageio import imread
from PIL import Image, ImageOps
import torchvision.models as models 
import torch.nn as nn
from matplotlib import pyplot as plt
import csv
import pandas as pd
import config
import PIL.Image as pilimg
from multiprocessing import Pool

 
class GANLoader(Dataset):
    
    def ChestXdataloader(self, img):

        # Adaptive masking
#         threshold = img.min() + (img.max() - img.min()) * 0.9
#         img[img > threshold] = 0

        # plt.imshow(img)
        # plt.show()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            normalize
        ])

        img = transform(img)

        return img

    def __init__(self):

        self.df1 = pd.read_csv('C:/Users/hb/Desktop/code/2.TF_to_Torch/Data/pggan_df.csv', index_col=0)
        # self.df2 = pd.read_csv('C:/Users/hb/Desktop/code/2.TF_to_Torch/Data/pggan2_df.csv', index_col=0)
        # self.df = pd.concat([self.df1, self.df2], ignore_index=True)
        self.df = self.df1.iloc[:, :]
        self.disease_cnt = [0]*14

        ### Need to be editted
        for i in range(len(self.df.index)):
            row = self.df.loc[i]
            for j in range(14):
                if row[j] == 1:
                    self.disease_cnt[j] += 1

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index): # 여기에 들어가는 index는?
        
        label = [0] * 14
        row = self.df.loc[index]
        img = cv2.imread(row[14])

        # plt.imshow(img)
        # plt.show()

        img = self.ChestXdataloader(img)
        for i in range(14):
            label[i] = row[i]
        label = torch.tensor(label)
        label = label.float()

        # print(label)

        return img, label

    def get_ds_cnt(self):
        return self.disease_cnt

class XRaysTestDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        # print('self.data_dir: ', self.data_dir)

        # full dataframe including train_val and test set
        self.df = self.get_df()

        self.make_pkl_dir(config.pkl_dir_path)

        # loading the classes list
        with open(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path), 'rb') as handle:
            self.all_classes = pickle.load(handle)
        

        # get test_df
        if not os.path.exists(os.path.join(config.pkl_dir_path, config.test_df_pkl_path)):

            self.test_df = self.get_test_df()
            # print('self.test_df.shape: ', self.test_df.shape)
            
            # pickle dump the test_df
            with open(os.path.join(config.pkl_dir_path, config.test_df_pkl_path), 'wb') as handle:
                pickle.dump(self.test_df, handle, protocol = pickle.HIGHEST_PROTOCOL)
            print('\n{}: dumped'.format(config.test_df_pkl_path))
        else:
            # pickle load the test_df
            with open(os.path.join(config.pkl_dir_path, config.test_df_pkl_path), 'rb') as handle:
                self.test_df = pickle.load(handle)
            # print('\n{}: loaded'.format(config.test_df_pkl_path))
            # print('self.test_df.shape: {}'.format(self.test_df.shape))

        self.disease_cnt = [0]*14
        self.all_classes = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation', 'No Finding']
        for i in range(len(self.test_df)):
            row = self.test_df.iloc[i, :]
            labels = str.split(row['Finding Labels'], '|')
            for lab in labels:  
                lab_idx = self.all_classes.index(lab)
                if lab_idx == 14: # No Finding
                    continue
                self.disease_cnt[lab_idx] += 1

    def get_ds_cnt(self):
        return self.disease_cnt

    def __getitem__(self, index):

        self.all_classes = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation', 'No Finding']

        row = self.test_df.iloc[index, :]
        
        img = cv2.imread(row['image_links'])
        labels = str.split(row['Finding Labels'], '|')
        
        target = torch.zeros(len(self.all_classes)) # 15
        new_target = torch.zeros(len(self.all_classes) - 1) # 14
        for lab in labels:
            lab_idx = self.all_classes.index(lab)
            target[lab_idx] = 1     

        # # Delete No Finding
        # i = self.all_classes.index('No Finding')
        # target = torch.cat([target[0:i], target[i+1:]])

        # #Change Label        
        # # Atelectasis : 0 -> 7
        # new_target[7] = target[0]
        # # Cardiomegaly : 1 -> 0
        # new_target[0] = target[1]
        # # Consolidation : 2 -> 13
        # new_target[13] = target[2]
        # # Edema : 3 -> 12
        # new_target[12] = target[3]
        # # Effusion : 4 -> 2
        # new_target[2] = target[4]
        # # Emphysema : 5 -> 1
        # new_target[1] = target[5]
        # # Fibrosis : 6 -> 11
        # new_target[11] = target[6]
        # # Hernia : 7 -> 3
        # new_target[3] = target[7]
        # # Infiltration : 8 -> 4
        # new_target[4] = target[8]
        # # Mass : 9 -> 5
        # new_target[5] = target[9]
        # # Nodule : 10 -> 6
        # new_target[6] = target[10]
        # # Pleural_Thickening : 11 -> 9
        # new_target[9] = target[11]
        # # Pneumonia : 12 -> 10
        # new_target[10] = target[12]
        # # Atelectasis : 13 -> 7
        # new_target[8] = target[13]

    
        if self.transform is not None:
            img = self.transform(img)
    
        return img, target[:14]

    def make_pkl_dir(self, pkl_dir_path):
        if not os.path.exists(pkl_dir_path):
            os.mkdir(pkl_dir_path)

    def get_df(self):
        csv_path = os.path.join(self.data_dir, 'Data_Entry_2017.csv')
        
        all_xray_df = pd.read_csv(csv_path)

        df = pd.DataFrame()        
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, 'images*', '*', '*.png'))]

        df['Image Index'] = df['image_links'].apply(lambda x : x[len(x)-16:len(x)])
        merged_df = df.merge(all_xray_df, how = 'inner', on = ['Image Index'])
        merged_df = merged_df[['image_links','Finding Labels']]
        return merged_df

    def get_test_df(self):

        # get the list of test data 
        test_list = self.get_test_list()

        test_df = pd.DataFrame()
        # print('\nbuilding test_df...')
        for i in tqdm(range(self.df.shape[0])):
            filename  = os.path.basename(self.df.iloc[i,0])
            # print('filename: ', filename)
            if filename in test_list:
                test_df = test_df.append(self.df.iloc[i:i+1, :])
         
        # print('test_df.shape: ', test_df.shape)

        return test_df

    def get_test_list(self):
        f = open( os.path.join('C:/Users/hb/Desktop/data/archive', 'test_list.txt'), 'r')
        test_list = str.split(f.read(), '\n')
        return test_list

    def __len__(self):
        return len(self.test_df)

class XRaysTrainDataset(Dataset):

    def __init__(self, data_dir, transform = None, indices=None):

        self.data_dir = data_dir
        self.extreme = False
        self.transform = transform
        # print('self.data_dir: ', self.data_dir)

        # full dataframe including train_val and test set
        self.df = self.get_df()
        self.df.to_csv('./Data/train_val_XRaysTrainDataset.csv')
        # print('self.df.shape: {}'.format(self.df.shape))

        self.make_pkl_dir(config.pkl_dir_path)
        # print(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path))
        # get train_val_df
        if not os.path.exists(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path)):

            self.train_val_df = self.get_train_val_df()
            # print('\nself.train_val_df.shape: {}'.format(self.train_val_df.shape))

            # pickle dump the train_val_df
            with open(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path), 'wb') as handle:
                pickle.dump(self.train_val_df, handle, protocol = pickle.HIGHEST_PROTOCOL)
            # print('{}: dumped'.format(config.train_val_df_pkl_path))
            
        else:
            # pickle load the train_val_df
            with open(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path), 'rb') as handle:
                self.train_val_df = pickle.load(handle)
            # print('\n{}: loaded'.format(config.train_val_df_pkl_path))
            # print('self.train_val_df.shape: {}'.format(self.train_val_df.shape))

        self.the_chosen, self.all_classes, self.all_classes_dict = self.choose_the_indices()
        ### sample indices
        self.the_chosen = indices
    
        if not os.path.exists(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path)):
            # pickle dump the classes list
            with open(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path), 'wb') as handle:
                pickle.dump(self.all_classes, handle, protocol = pickle.HIGHEST_PROTOCOL)
                print('\n{}: dumped'.format(config.disease_classes_pkl_path))
        else:
            pass
            # print('\n{}: already exists'.format(config.disease_classes_pkl_path))

        self.new_df = self.train_val_df.iloc[self.the_chosen, :] # this is the sampled train_val data
        self.new_df = self.new_df.reset_index()
        # Do not sample the training data
        # self.new_df = self.train_val_df

        self.disease_cnt = [0]*14
        self.all_classes = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation', 'No Finding']
        delete_lst = []
        if self.extreme == True: # extreme case
            for i in range(len(self.new_df)):
                row = self.new_df.iloc[i, :]
                labels = str.split(row['Finding Labels'], '|')
                for lab in labels:  
                    lab_idx = self.all_classes.index(lab)
                    if lab_idx == 14: # No Finding
                        continue
                    if lab_idx == 2 or lab_idx == 7:
                        if self.disease_cnt[lab_idx] > 1:
                            delete_lst.append(i)
                            break
                    self.disease_cnt[lab_idx] += 1
            self.new_df = self.new_df.drop(index=delete_lst, axis=0)
            self.disease_cnt = [0]*14

        for i in range(len(self.new_df)):
            row = self.new_df.iloc[i, :]
            labels = str.split(row['Finding Labels'], '|')
            for lab in labels:  
                lab_idx = self.all_classes.index(lab)
                if lab_idx == 14: # No Finding
                    continue
                self.disease_cnt[lab_idx] += 1

    def get_ds_cnt(self):
        return self.disease_cnt

        # print('\nself.all_classes_dict: {}'.format(self.all_classes_dict))
            
    def compute_class_freqs(self):
        """
        Compute positive and negative frequences for each class.

        Args:
            labels (np.array): matrix of labels, size (num_examples, num_classes)
        Returns:
            positive_frequencies (np.array): array of positive frequences for each
                                          class, size (num_classes)
            negative_frequencies (np.array): array of negative frequences for each
                                          class, size (num_classes)
        """    
        # total number of patients (rows)
        labels = self.train_val_df ## What is the shape of this???
        N = labels.shape[0]
        positive_frequencies = (labels.sum(axis = 0))/N
        negative_frequencies = 1.0 - positive_frequencies
    
        return positive_frequencies, negative_frequencies

    def resample(self):
        self.the_chosen, self.all_classes, self.all_classes_dict = self.choose_the_indices()
        # self.new_df = self.train_val_df.iloc[self.the_chosen, :]
        self.new_df = self.train_val_df

        # print('\nself.all_classes_dict: {}'.format(self.all_classes_dict))

    def make_pkl_dir(self, pkl_dir_path):
        if not os.path.exists(pkl_dir_path):
            os.mkdir(pkl_dir_path)

    def get_train_val_df(self):

        # get the list of train_val data 
        train_val_list = self.get_train_val_list()
        print("train_va_list: ",len(train_val_list))

        train_val_df = pd.DataFrame()
        print('\nbuilding train_val_df...')
        for i in tqdm(range(self.df.shape[0])):
            filename  = os.path.basename(self.df.iloc[i,0])
            # print('filename: ', filename)
            if filename in train_val_list:
                train_val_df = train_val_df.append(self.df.iloc[i:i+1, :])

        # print('train_val_df.shape: {}'.format(train_val_df.shape))

        return train_val_df

    def __getitem__(self, index):

        self.all_classes = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation', 'No Finding']

        row = self.new_df.iloc[index, :]

        img = cv2.imread(row['image_links'])
        labels = str.split(row['Finding Labels'], '|')
        
        target = torch.zeros(len(self.all_classes))
        new_target = torch.zeros(len(self.all_classes) - 1)
        for lab in labels:
            lab_idx = self.all_classes.index(lab)
            target[lab_idx] = 1            
    
        if self.transform is not None:
            img = self.transform(img)

        #         # Delete No Finding
        # i = self.all_classes.index('No Finding')
        # target = torch.cat([target[0:i], target[i+1:]])

        # #Change Label        
        # # Atelectasis : 0 -> 7
        # new_target[7] = target[0]
        # # Cardiomegaly : 1 -> 0
        # new_target[0] = target[1]
        # # Consolidation : 2 -> 13
        # new_target[13] = target[2]
        # # Edema : 3 -> 12
        # new_target[12] = target[3]
        # # Effusion : 4 -> 2
        # new_target[2] = target[4]
        # # Emphysema : 5 -> 1
        # new_target[1] = target[5]
        # # Fibrosis : 6 -> 11
        # new_target[11] = target[6]
        # # Hernia : 7 -> 3
        # new_target[3] = target[7]
        # # Infiltration : 8 -> 4
        # new_target[4] = target[8]
        # # Mass : 9 -> 5
        # new_target[5] = target[9]
        # # Nodule : 10 -> 6
        # new_target[6] = target[10]
        # # Pleural_Thickening : 11 -> 9
        # new_target[9] = target[11]
        # # Pneumonia : 12 -> 10
        # new_target[10] = target[12]
        # # Atelectasis : 13 -> 7
        # new_target[8] = target[13]
    
        return img, target[:14]
        
    def choose_the_indices(self):
        
        max_examples_per_class = 10000 # its the maximum number of examples that would be sampled in the training set for any class
        the_chosen = []
        all_classes = {}
        length = len(self.train_val_df)
        # for i in tqdm(range(len(merged_df))):
        # print('\nSampling the huuuge training dataset')
        for i in list(np.random.choice(range(length),length, replace = False)):
            
            temp = str.split(self.train_val_df.iloc[i, :]['Finding Labels'], '|')

            # special case of ultra minority hernia. we will use all the images with 'Hernia' tagged in them.
            if 'Hernia' in temp:
                the_chosen.append(i)
                for t in temp:
                    if t not in all_classes:
                        all_classes[t] = 1
                    else:
                        all_classes[t] += 1
                continue

            # choose if multiple labels
            if len(temp) > 1:
                bool_lis = [False]*len(temp)
                # check if any label crosses the upper limit
                for idx, t in enumerate(temp):
                    if t in all_classes:
                        if all_classes[t]< max_examples_per_class: # 500
                            bool_lis[idx] = True
                    else:
                        bool_lis[idx] = True
                # if all lables under upper limit, append
                if sum(bool_lis) == len(temp):                    
                    the_chosen.append(i)
                    # maintain count
                    for t in temp:
                        if t not in all_classes:
                            all_classes[t] = 1
                        else:
                            all_classes[t] += 1
            else:        # these are single label images
                for t in temp:
                    if t not in all_classes:
                        all_classes[t] = 1
                    else:
                        if all_classes[t] < max_examples_per_class: # 500
                            all_classes[t] += 1
                            the_chosen.append(i)

        # print('len(all_classes): ', len(all_classes))
        # print('all_classes: ', all_classes)
        # print('len(the_chosen): ', len(the_chosen))
        
        '''
        if len(the_chosen) != len(set(the_chosen)):
            print('\nGadbad !!!')
            print('and the difference is: ', len(the_chosen) - len(set(the_chosen)))
        else:
            print('\nGood')
        '''

        return the_chosen, sorted(list(all_classes)), all_classes
    
    def get_df(self):
        csv_path = os.path.join(self.data_dir, 'Data_Entry_2017.csv')
        # print('\n{} found: {}'.format(csv_path, os.path.exists(csv_path)))
        
        all_xray_df = pd.read_csv(csv_path)

        df = pd.DataFrame()        
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, 'images*', '*', '*.png'))]

        df['Image Index'] = df['image_links'].apply(lambda x : x[len(x)-16:len(x)])
        merged_df = df.merge(all_xray_df, how = 'inner', on = ['Image Index'])
        merged_df = merged_df[['image_links','Finding Labels']]
        return merged_df
    
    def get_train_val_list(self):
        f = open("C:/Users/hb/Desktop/data/archive/train_val_list.txt", 'r')
        train_val_list = str.split(f.read(), '\n')
        return train_val_list

    def __len__(self):
        return len(self.new_df)

class ChexpertTrainDataset(Dataset):

    

    def __init__(self, transform = None, indices = None):
        
        csv_path = "C:/Users/hb/Desktop/data/CheXpert-v1.0-small/train.csv"
        self.dir = "C:/Users/hb/Desktop/data/"
        self.transform = transform

        self.all_data = pd.read_csv(csv_path)
        self.all_data = self.all_data.drop(columns=['Sex','Age','AP/PA'])
        self.all_data = self.all_data.drop(columns=['Support Devices', 'No Finding'])
        # 'Pleural Effusion','Pleural Other',
        self.all_data = self.all_data.fillna(0)
        self.all_data = self.all_data.replace(-1,1) # U1
        
        self.frontal_data = self.all_data[self.all_data['Frontal/Lateral'] == 'Frontal']
        self.selecte_data = self.all_data.iloc[indices, :]
        # self.selecte_data.to_csv("C:/Users/hb/Desktop/data/CheXpert-v1.0-small/selected_data.csv")
        self.class_num = 12
        
        
    def get_ds_cnt(self):
        total_ds_cnt = [0] * self.class_num
        for i in range(len(self.selecte_data)):
            row = self.selecte_data.iloc[i, 2:]
            for j in range(len(row)):
                total_ds_cnt[j] += int(row[j])
        return total_ds_cnt

    def __getitem__(self, index):

        row = self.selecte_data.iloc[index, :]
        # img = cv2.imread(self.dir + row['Path'])
        img = pilimg.open(self.dir + row['Path'])
        label = torch.FloatTensor(row[2:])
        gray_img = self.transform(img)
        return torch.cat([gray_img,gray_img,gray_img], dim = 0), label

    def __len__(self):
        return len(self.selecte_data)

class ChexpertTestDataset(Dataset):

    def __init__(self, transform = None):
        
        csv_path = "C:/Users/hb/Desktop/data/CheXpert-v1.0-small/valid.csv"
        self.dir = "C:/Users/hb/Desktop/data/"
        self.transform = transform

        self.all_data = pd.read_csv(csv_path)
        self.all_data = self.all_data.drop(columns=['Sex','Age','AP/PA'])
        self.all_data = self.all_data.drop(columns=['Support Devices', 'No Finding'])
        # ,'Pleural Effusion','Pleural Other'
        self.all_data = self.all_data.fillna(0)
        self.all_data = self.all_data.replace(-1,1)
        
        self.frontal_data = self.all_data[self.all_data['Frontal/Lateral'] == 'Frontal']
        self.selecte_data = self.all_data.iloc[:, :]
        # self.selecte_data.to_csv("C:/Users/hb/Desktop/data/CheXpert-v1.0-small/selected_data.csv")
        self.class_num = 12

    def __getitem__(self, index):

        row = self.selecte_data.iloc[index, :]
        img = pilimg.open(self.dir + row['Path'])
        label = torch.FloatTensor(row[2:])
        gray_img = self.transform(img)

        return torch.cat([gray_img,gray_img,gray_img], dim = 0), label

    def get_ds_cnt(self):
        total_ds_cnt = [0] * self.class_num
        for i in range(len(self.selecte_data)):
            row = self.selecte_data.iloc[i, 2:]
            for j in range(len(row)):
                total_ds_cnt[j] += int(row[j])
        return total_ds_cnt

    def __len__(self):
        return len(self.selecte_data)


class CIFAR10TrainDataset(Dataset):

    def __init__(self):
        path = 'C:/Users/hb/Desktop/data/CIFAR10_Client_random/train.csv'
        self.data = pd.read_csv(path)

        self.transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ])

    def __getitem__(self, index):
        
        row = self.data.iloc[index, :]
        img = pilimg.open(row['path'])
        label = torch.FloatTensor(row[2:])
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)

class CIFAR10TestDataset(Dataset):

    def __init__(self):
        path = 'C:/Users/hb/Desktop/data/CIFAR10_Client_random/test.csv'
        self.data = pd.read_csv(path)

        self.transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ])

    def __getitem__(self, index):
        
        row = self.data.iloc[index, :]
        img = pilimg.open(row['path'])
        label = torch.FloatTensor(row[2:])
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)