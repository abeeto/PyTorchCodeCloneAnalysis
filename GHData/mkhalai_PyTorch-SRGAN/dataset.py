
from config import IMAGE_PATH
import os
from transforms import preprocess_train
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image

class TrainDataset(Dataset):
    def __init__(self, directory = IMAGE_PATH):
        super(TrainDataset, self).__init__()
        self.files = [os.path.join(directory, x) for x in os.listdir(directory)]

    def __getitem__(self, index):
        im = read_image(self.files[index])
        return preprocess_train(im)

    def __len__(self):
        return len(self.files)
