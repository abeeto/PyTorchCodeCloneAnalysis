from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import torch
class MyData(Dataset):
    def __init__(self,dirs,transforms=None):
        super(MyData,self).__init__()
        images=[]
        for d in os.listdir(dirs):
            files=dirs+str(d)
            if os.path.isdir(files)==False:
                continue
            for f in os.listdir(files):
                if f.endswith(('png','bmp','jpg','gif','jpeg')):
                    images.append(files+"/"+f)
        self.images=images
        self.transforms=transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        files=self.images[index]
        label=int(files.split('/')[-2])
        image=Image.open(files).convert('RGB')
        if self.transforms is not None:
            image=self.transforms(image)
        return image,label

def collate_fn(images):
    label=[]
    image=[]
    for i,data in enumerate(images):
        image.append(data[0])
        label.append(data[1])
    return torch.stack(image,0),torch.from_numpy(np.array(label,dtype='int'))
