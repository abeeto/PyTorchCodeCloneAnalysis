import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import os
from ast import literal_eval
import numpy as np
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


class CustomDataset(Dataset):
    def __init__(self, args):
        
        self.args = args

        self.dataset_path = args.dataset_path
        self.img_height = args.img_height
        self.img_width = args.img_width
        self.channels = args.img_ch
        self.segmap_channel = args.segmap_ch
        self.augment_flag = args.augment_flag # Image augmentation use or not
        self.batch_size = args.batch_size

        self.img_dataset_path = os.path.join(self.dataset_path, 'image')
        self.segmap_dataset_path = os.path.join(self.dataset_path, 'segmap')
        # self.segmap_test_dataset_path = os.path.join(dataset_path, 'segmap_test')

        self.image = glob(self.img_dataset_path + '/*.*')
        self.segmap = glob(self.segmap_dataset_path + '/*.*')
        self.segmap_label_path = os.path.join(self.dataset_path, 'segmap_label.txt')
        # self.segmap_test = []
        self.color_value_dict = {}

        self.preprocess()

    def preprocess(self):

        # semap_label.txt 가 존재하는 경우
        if os.path.exists(self.segmap_label_path):
            print("segmap_label exists ! ")

            with open(self.segmap_label_path, 'r') as f:
                self.color_value_dict = literal_eval(f.read())
                print(self.color_value_dict)

        # semap_label.txt 가 존재하지 않는 경우
        else: 
            print("segmap_label no exists ! ")
            label = 0
            for img in tqdm(self.segmap) :

                x = Image.open(img).convert('RGB')
                    
                x = x.resize((self.img_height, self.img_width), Image.NEAREST)
                x = x

                h, w, c = np.array(x).shape

                for i in range(h) :
                    for j in range(w) :
                        if tuple(x.getpixel((i, j))) not in self.color_value_dict.keys(): 
                            self.color_value_dict[tuple(x.getpixel((i, j)))] = label
                            label += 1
                
            print('len of color_value_dict is ', len(self.color_value_dict))
            print(self.color_value_dict)
            with open(self.segmap_label_path, 'w') as f :
                f.write(str(self.color_value_dict))

    def convert_from_color_segmentation(self, color_value_dict, arr_3d):
        arr_3d = arr_3d.resize((self.img_height, self.img_width), Image.NEAREST)
        arr_2d = torch.Tensor(self.img_height, self.img_width)
        
        for i in range(self.img_height):
            for j in range(self.img_width):
                arr_3d_color = tuple(arr_3d.getpixel((i, j)))
                if arr_3d_color in self.color_value_dict.keys():
                    # print("yes!")
                    arr_2d[i][j] = self.color_value_dict[arr_3d_color]
        
        return arr_2d

    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        
        image_transformer = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        segmap = Image.open(self.segmap[idx])
        segmap_tensor = self.convert_from_color_segmentation(self.color_value_dict, segmap) # 리턴하면서 텐서가 되었다.
        # print(segmap_tensor.shape) torch.Size([256, 256])
        segmap_tensor = segmap_tensor.unsqueeze(0) # 배치 차원 추가

        image = Image.open(self.image[idx])
        image = image.convert('RGB')
        image_tensor = image_transformer(image)
        return image_tensor, segmap_tensor

def imsave(input, name):
    input = input.numpy().transpose((1, 2, 0))
    plt.imshow(input)
    plt.savefig(name)

def load_dataset(args):
    
    train_datasets = CustomDataset(args)
    # train_datasets.__getitem__(0)
    train_dataloader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=False, drop_last=True)
    # train_dataloader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=False)
    
    return train_dataloader

# if __name__ == "__main__":
    # from main import *
    # args = parse_args()
    
    # load_dataset(args)
