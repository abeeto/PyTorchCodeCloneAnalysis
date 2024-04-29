from __future__ import division
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pdb
import cv2
import torch
from PIL import Image
from skimage import io
import skimage
import math

import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resolution = 64
sdf_dir_base = '/local-scratch/mma/DISN/ShapeNetOut64/03001627'
render_dir_base = '/local-scratch/mma/DISN/ShapeNetRendering/03001627'

class ImageDataset(Dataset):
    def __init__(self, dir, phase):
        super(ImageDataset, self).__init__()

        if phase not in ['train', 'test', 'val', 'val1', 'b1', 'c1', 'd1']: #remove val1
            raise ValueError('invalid phase {}'.format(phase))

        self.phase = phase
        self.render_dir = []
        self.sdf_dir = []

        phase_dir = os.path.join(dir, phase + '.txt')
        with open( phase_dir, 'rb') as f:
            #lines = f.readlines()
            lines = f.read().splitlines()

        for i in range(len(lines)):
            path1 = os.path.join(render_dir_base, lines[i])
            self.render_dir.append(path1)

            path2 = os.path.join(sdf_dir_base, lines[i])
            self.sdf_dir.append(path2)

    def __len__(self):
        return len(self.render_dir)

    def __getitem__(self, idx):

        render_path = self.render_dir[idx]
        sdf_path = self.sdf_dir[idx]

        with open(render_path, 'rb') as f:

            img = cv2.imread(render_path)
            img1 = cv2.resize(img, (224,224))

        img1 = skimage.img_as_float(img1)
        resized_rendered_img = np.transpose(img1, (2, 0, 1))

        path_split = render_path.split(os.sep)



        #camera parameters for inference
        split1 = path_split[:-1]
        metadata_path = os.path.join(*split1)
        metadata_path = '/'+ metadata_path
        metadata_path = os.path.join(metadata_path, 'rendering_metadata.txt')

        index = int(path_split[-1][:-4])
        with open(metadata_path, 'rb') as f:
            lines = f.read().splitlines()
        camera_param = lines[index]
        camera_param =camera_param.split()
        c = [float(i) for i in camera_param]
        camera_param = torch.FloatTensor(c)

        path_split = sdf_path.split(os.sep)
        split = path_split[:-2]
        sdf_path2 = os.path.join(*split)
        sdf_path2 = '/'+ sdf_path2
        sdf_path2 = os.path.join(sdf_path2, 'SDF.npy') 


        read_dict = np.load(sdf_path2, allow_pickle=True)
        #SDF_dict = read_dict.item().get('SDF')
        b_min = read_dict.item().get('bmin')
        b_max = read_dict.item().get('bmax')



        index = path_split[-1][:-4]

        path_split = sdf_path.split(os.sep)
        split = path_split[:-2]
        sdf_path = os.path.join(*split)
        sdf_path = '/'+ sdf_path

        sdf_path = sdf_path + '/' + str(index) + '.npy' 

        read_dict = np.load(sdf_path, allow_pickle=True)

        sdf = read_dict.item().get('sdf') 

        proj_point = read_dict.item().get('proj_point')
        point = read_dict.item().get('point')



        data = {'image': resized_rendered_img, 'sdf': sdf, 'point': point, 'proj_point': proj_point, 'camera_param': camera_param, 'b_min': b_min, 'b_max': b_max}
        return data





if __name__ == '__main__':

    dir = '/local-scratch/mma/DISN/version1'
    dataset1 = ImageDataset(dir, phase='val1')
    data_loader1 = DataLoader(dataset1, batch_size=1, shuffle=True)

    print("HI")
    # Check DataLoader

    for iter_i, batch_data in enumerate(data_loader1):
        print("iter:", iter_i)
        image = batch_data['image']
        sdf = batch_data['sdf']
        point = batch_data['point']
        proj_point = batch_data['proj_point']
        camera_param = batch_data['camera_param']
        b_min = batch_data['b_min']
        b_max = batch_data['b_max']

        print(image.shape)
        print(point.shape)
        print(proj_point.shape)
        print(sdf.shape)
        print(camera_param)
        print(b_min)
        print(b_max)



    '''
    # Check DataLoader

    for iter_i, batch_data in enumerate(data_loader1):
        print("iter:", iter_i)
        image = batch_data['image']
        camera_param = batch_data['camera_param']
        path = batch_data['path']

        print(camera_param.shape)
        print(image.shape)
        print(len(path))
        #print(camera_param)


        for j in range(16):
            dataset2 = PointDataset(path[j])
            data_loader2 = DataLoader(dataset2, batch_size=2048, shuffle=True)

            p = next(iter(data_loader2))
            #print(p['point'])
            #print(p['point'].shape)

            for iter_i2, batch_data2 in enumerate(data_loader2):
                print("iter2:", iter_i2)
                bmin = batch_data2['b_min']
                point = batch_data2['point']
                point1 = batch_data2['point1']
                sdf = batch_data2['sdf']
                path2 = batch_data2['path']
                print(point.shape)

    '''
