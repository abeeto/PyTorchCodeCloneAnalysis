from PIL import Image
from torch.utils.data import DataLoader,Dataset

class ATTDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        img_path, target = self.imageFolderDataset.imgs[index]

        img = Image.open(img_path)
        img = img.convert("L")

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)