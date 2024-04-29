import glob
import cv2
from torchvision import *
from torch.utils.data import Dataset, DataLoader
train_data_path='Potato\Train'
val_data_path='Potato\Valid'
test_data_path='Potato\Test'
class customdata(Dataset):
    def __init__(self,path,transform=None):
        self.data_paths=glob.glob(path + '/*')
        self.image_paths=glob.glob(path+ '**/*' + '/*')
        self.transform=transform
    def image (self):
        classes = []
        for data_path in self.data_paths:
            classes.append(data_path.split('\\')[-1])
        class_to_idx = {j: i for i, j in enumerate(classes)}
        return class_to_idx
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image_paths=self.image_paths
        class_to_idx=self.image()
        image_filepath =image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = image_filepath.split('\\')[-2]
        label = class_to_idx[label]
        if self.transform:
            image = self.transform(image)
        return image, label
def loader(image_path,transform=None,batch=32,shuffle=True):
    test_dataset = customdata(image_path, transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch, shuffle=shuffle)
    return test_loader
