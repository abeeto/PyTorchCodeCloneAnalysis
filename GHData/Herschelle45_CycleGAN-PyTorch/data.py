from torch.utils.data import Dataset 
import numpy as np 
from PIL import Image 
import os 

class HorseZebraDataset(Dataset):
    def __init__(self, horsedir, zebradir, transform=None):
        self.horsedir = horsedir
        self.zebradir = zebradir
        self.transform = transform 
        self.dslen =  max(len(self.horsedir), len(zebradir))
    def __len__(self):
        return self.dslen 
    def __getitem__(self, index):
        horsefiles = os.listdir(self.horsedir)
        zebrafiles = os.listdir(self.zebradir)
        horse = np.array(Image.open(os.path.join(self.horsedir, horsefiles[index % self.dslen])).convert('RGB'))
        zebra = np.array(Image.open(os.path.join(self.zebradir, zebrafiles[index % self.dslen])).convert('RGB'))
        if self.transform:
            horse = self.transform(horse)
            zebra = self.transform(zebra)
        return horse, zebra 
