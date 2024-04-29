import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class dataset(Dataset):
    def __init__(self,img_dir,mask_dir,img_names,transform=None):
        self.img_dir=img_dir
        self.mask_diir=mask_dir
        self.transform=transform
        self.img_names=img_names #Train/val img_names
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self,idx):
        img_path=os.path.join(self.img_dir,self.img_names[idx])
        mask_path=os.path.join(self.mask_diir,self.img_names[idx].replace(".jpg","_mask.gif"))
        img=np.array(Image.open(img_path).convert("RGB"),dtype=np.float32)
        mask=np.array(Image.open(mask_path).convert("L"),dtype=np.float32) #make sure mask image is gray image
        mask[mask==255.0]=1.0

        if self.transform:
            augmentation=self.transform(image=img,mask=mask)
            image=augmentation["image"]
            mask=augmentation["mask"]
        return image,mask