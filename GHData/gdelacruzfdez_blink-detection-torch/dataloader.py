import os

import numpy as np
import numpy.random as rand
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler
import pandas as pd
from PIL import Image
from evaluation import evaluate
from functools import reduce

POSITIVE_NEGATIVE_RATIO = 0.5
SAME_CLASS = 1

BLINK_DETECTION_MODE = 'BLINK_DETECTION_MODE'
BLINK_COMPLETENESS_MODE = 'BLINK_COMPLETENESS_MODE'
EYE_STATE_DETECTION_MODE = 'EYE_STATE_DETECTION_MODE'

class BlinkDataset(Dataset):

    def __init__(self, paths, transform,y_col='target', videos = None):
        self.x_col = 'complete_path'
        self.y_col = y_col
        self.transform = transform

        self.__initialize_dataframe_from_paths(paths, videos)
        self.set_target_column()
        self.targets = self.dataframe[self.y_col]
        self.classes = np.unique(self.dataframe[self.y_col])
        self.dataframe['pred'] = 0

    def __initialize_dataframe_from_paths(self, paths, videos):
        dataframes = []
        max_num_video = 0
        for root in paths:
            csvFilePath = root + '.csv'
            dataframe = pd.read_csv(csvFilePath)
            dataframe['base_path'] = root

            completePaths = []
            for idx, row in dataframe.iterrows():
                completePaths.append(os.path.join(row['base_path'], row['frame']))
            dataframe['complete_path'] = completePaths
            if videos != None:
                dataframe = dataframe[dataframe.video.isin(videos)]
            dataframe_max_video = dataframe['video'].max()
            dataframe['video'] = dataframe['video'] +  max_num_video
            max_num_video = dataframe_max_video
            dataframes.append(dataframe)
    
        self.dataframe = pd.concat(dataframes, ignore_index=True, sort=False)
        self.dataframe = self.dataframe.rename_axis('idx').sort_values(by=['eye', 'idx'], ascending=[True, True]).reset_index()
    
    def __len__(self):
        return len(self.dataframe)

    def getDataframeRow(self, idx):
        return self.dataframe.iloc[idx]

    def getDataframe(self):
        return self.dataframe
    
    def set_target_column(self):
        # Must be overrriden in child classes
        pass

from torchvision.transforms import transforms
basic_transform = transforms.Compose([
    transforms.Resize((100, 100), interpolation=Image.BICUBIC),
    transforms.ToTensor()
    ])

class LSTMDataset(BlinkDataset):

    def __init__(self, paths, transform, videos = None):
        super().__init__(paths, transform, videos=videos)


    def __len__(self):
        return len(self.dataframe) 

    def getDataframeRow(self, idx):
        return self.dataframe.iloc[idx]

    def getDataframe(self):
        return self.dataframe
    
    def __getitem__(self, idx):
        selectedRow = self.dataframe.iloc[idx]
        if 'NOT_VISIBLE' in selectedRow['complete_path']:
            sample = Image.new('RGB', (64, 64))
        else:
            sample = Image.open(selectedRow['complete_path']).convert('RGB')
        not_transformed = basic_transform(sample)
        target = selectedRow[self.y_col]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, not_transformed
        

class BlinkDetectionLSTMDataset(LSTMDataset):

    def set_target_column(self):
        self.dataframe[self.y_col] = (self.dataframe['blink_id'].astype(int) > 0).astype(int)


class BlinkCompletenessDetectionLSTMDataset(LSTMDataset):

    def set_target_column(self):
        self.dataframe[self.y_col] = (self.dataframe['blink_id'].astype(int) > 0).astype(int) + self.dataframe['blink'].astype(int)


class EyeStateDetectionSingleInputLSTMDataset(LSTMDataset):

    def set_target_column(self):
        self.dataframe[self.y_col] = (self.dataframe['blink'] > 0).astype(int)


class EyeStateDetectionLSTMDataset(Dataset):

    def __init__(self, paths, transform,  videos = None):
        self.x_col = 'complete_path'
        self.y_col = 'target'
        self.transform = transform

        dataframes = []
        max_num_video = 0
        for root in paths:
            csvFilePath = root + '.csv'
            dataframe = pd.read_csv(csvFilePath)
            dataframe['base_path'] = root

            completePaths = []
            for idx, row in dataframe.iterrows():
                completePaths.append(os.path.join(row['base_path'], row['frame']))
            dataframe['complete_path'] = completePaths
            if videos != None:
                dataframe = dataframe[dataframe.video.isin(videos)]
            dataframe_max_video = dataframe['video'].max()
            dataframe['video'] = dataframe['video'] +  max_num_video
            max_num_video = dataframe_max_video
            dataframes.append(dataframe)
        
        self.dataframe = pd.concat(dataframes, ignore_index=True, sort=False)
        self.dataframe = self.dataframe.rename_axis('idx').sort_values(by=['eye', 'idx'], ascending=[True, True]).reset_index()

        self.left_eyes = self.dataframe[self.dataframe['eye'] == 'LEFT']
        self.right_eyes = self.dataframe[self.dataframe['eye'] == 'RIGHT']
        #blinks_per_frame = self.dataframe.groupby(['base_path','video','frameId'])
        self.dataframe = self.right_eyes
        #self.targets = blinks_per_frame.blink.apply(lambda x: reduce(lambda a,b: a*b ,x.values.tolist())).values
        #self.dataframe['targets'] = self.targets
        self.targets = (self.left_eyes.blink.values * self.right_eyes.blink.values).astype(int)
        #self.dataframe.to_csv('targets.csv')
        print(sum(self.targets))
        print('lens',len(self.left_eyes), len(self.right_eyes), len(self.targets))


    def __len__(self):
        return len(self.targets) 

    def getDataframeRow(self, idx):
        return self.dataframe.iloc[idx]

    def getDataframe(self):
        return self.dataframe

    def __get_eye_image(self, idx, position):
        if 'RIGHT' == position:
            eye_row = self.right_eyes.iloc[idx]
        else:
            eye_row = self.left_eyes.iloc[idx]

        if 'NOT_VISIBLE' in eye_row['complete_path']:
            eye_image =  Image.new('RGB', (100,100))
        else:
            eye_image = Image.open(eye_row['complete_path']).convert('RGB')
        
        return eye_image
    
    def __getitem__(self, idx):
        left_eye = self.__get_eye_image(idx, 'LEFT')
        right_eye = self.__get_eye_image(idx, 'RIGHT')
        
        target = self.targets[idx]

        if self.transform is not None:
            left_eye = self.transform(left_eye)
            right_eye = self.transform(right_eye)

        return (left_eye, right_eye), target


class SiameseDataset(BlinkDataset):

    def __init__(self, paths, transform, videos = None ):
        super().__init__(paths, transform,y_col='blink', videos=videos)
        self.targets = self.dataframe[self.y_col]
        self.classes = np.unique(self.dataframe[self.y_col])
    
    def set_target_column(self):
        self.dataframe[self.y_col] = (self.dataframe['blink_id'].astype(int) > 0).astype(int) + self.dataframe['blink'].astype(int)
        #self.dataframe[self.y_col] = self.dataframe['blink'].astype(int)

    
    def __getitem__(self, idx):
        target = int(rand.random_sample() > POSITIVE_NEGATIVE_RATIO)

        y = self.dataframe[self.y_col].to_numpy().astype(int)

        class1 = rand.choice(self.classes)
        if target == SAME_CLASS:
            class2 = class1
        else:
            class2 = rand.choice(list((set(self.classes) - {class1})))

        idx1 = rand.choice(np.argwhere(y == class1).flatten())
        selectedRow1 = self.dataframe.iloc[idx1]

        idx2 = rand.choice(np.argwhere(y == class2).flatten())
        selectedRow2 = self.dataframe.iloc[idx2]

        sample1 = Image.open(selectedRow1['complete_path']).convert('RGB')
        sample2 = Image.open(selectedRow2['complete_path']).convert('RGB')


        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        return (sample1, sample2), target


class BalancedBatchSampler(BatchSampler):

    def __init__(self, targets, n_classes, n_samples):
        self.targets = targets
        self.classes = list(set(self.targets))
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.targets) 
        self.batch_size = self.n_classes * self.n_samples

        self.target_to_idxs = {target: np.where(np.array(self.targets) == target)[0] for target in self.classes}
    
    def __iter__(self):
        count = 0
        while count + self.batch_size < self.n_dataset: # * 20
            indices = []
            for target in np.random.choice(self.classes, self.n_classes, replace = False):
                indices.extend(np.random.choice(self.target_to_idxs[target], self.n_samples, replace=False))
            yield indices
            count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size

        

