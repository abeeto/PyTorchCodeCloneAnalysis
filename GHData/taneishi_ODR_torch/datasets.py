import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os

class ODIR5K(Dataset):
    def __init__(self, phase, transform=None):
        super(ODIR5K, self).__init__()
        self.csvfile = 'labels/%s.csv' % (phase)
        self.df = pd.read_csv(self.csvfile)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index, crop=True):
        row = self.df.iloc[index]

        image = 'images/%s' % (row['ID'])
        if os.path.exists(image):
            image = Image.open(image).convert('RGB')
        else:
            print('%s not found' % (image))

        label = torch.FloatTensor([int(i) for i in row['Normal':'Others']])

        if crop: # crop black pixels
            image = np.array(image)

            # Mask of coloured pixels.
            mask = image > 0

            # Coordinates of coloured pixels.
            coordinates = np.argwhere(mask)

            # Binding box of non-black pixels.
            x0, y0, s0 = coordinates.min(axis=0)
            x1, y1, s1 = coordinates.max(axis=0) + 1 # slices are exclusive at the top.

            # Get the contents of the bounding box.
            image = image[x0:x1, y0:y1]
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
        ])
    
    dataset = ODIR5K('train', transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for image, label in dataloader:
        image = image.view(3, 500, 500).permute(1, 2, 0)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.tight_layout()
        plt.savefig('figure/input.png')

        break
