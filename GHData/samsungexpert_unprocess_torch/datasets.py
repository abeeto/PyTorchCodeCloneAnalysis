import glob, random, os
from PIL import Image
from torch.utils.data import DataLoader
import torch
from test_unprocess import unprocess
from torchvision.transforms import ToTensor, ToPILImage

torch.set_num_threads(4)
class Dataset(DataLoader):
    def __init__(self, dataset_dir, transforms):
        self.dataset_dir = dataset_dir

        folder = glob.glob(self.dataset_dir+"/**/**/")
        filepath =[]

        for temp in folder:
            folder_num = temp.split('/')[-2]
            if int(folder_num) % 20 == 1:
                print(temp)
                filepath += glob.glob(temp+"/**/**.png",recursive=True)

        self.image_path=filepath
        self.transform = transforms

    def __getitem__(self, idx ):
        index= random.randint(0, len(self.image_path) - 1)

        item = Image.open(self.image_path[index]).convert('RGB')
        item_tensor = ToTensor()(item)

        item_tensor = unprocess(item_tensor)
        item = self.transform(ToPILImage()(item_tensor))

        return item

    def __len__(self):
        return len(self.image_path)

