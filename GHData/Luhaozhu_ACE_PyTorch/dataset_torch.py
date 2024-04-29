from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import random
from PIL import Image
import pandas as pd


class ImageDataset(Dataset):
    """construct image dataset"""
    def __init__(self,source_dir,target_class,label_path,max_imgs,transform=True):
        super(ImageDataset).__init__()
        label_csv = pd.read_csv(label_path,index_col='label_name')
        self.target_class = target_class

        file_dir_name = label_csv.loc[self.target_class][0]
        self.label_index = label_csv.loc[self.target_class][1]
        self.source_dir = os.path.join(source_dir,file_dir_name)
        
        self.transform = transform
        self.max_imgs = max_imgs
        
        if self.transform:
            self.transform_fn = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),  # 变成Tensor格式，归一化到[0,1区间]
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化均值和方差，与ImageNet的均值和方差一致
            ])
        all_file = os.listdir(self.source_dir)
        self.random_img_list = random.sample(all_file,self.max_imgs)

    def __len__(self):
        return self.max_imgs
    def __getitem__(self, index):
        img_dir = os.path.join(self.source_dir,self.random_img_list[index])
        img = Image.open(img_dir).convert('RGB')
        if self.transform:
            img = self.transform_fn(img)
        return img,self.label_index


# if __name__ == "__main__":

#     source_dir = "/data/dataset/ImageNet2012/train"
#     target_class = "zebra"
#     label_path = "/data/dataset/ImageNet2012/imagenet_labels.csv"
#     max_imgs = 40
#     image_dataset = ImageDataset(source_dir,target_class,label_path,max_imgs)
#     dataloader = DataLoader(image_dataset,batch_size=4,shuffle=True)
#     for x,y in dataloader:
#         print(x.shape)
#         print(y)
    