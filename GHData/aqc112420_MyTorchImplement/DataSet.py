from scipy.misc import imread
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MyDataset(Dataset):
    def __init__(self, base_dir=r'E:\anqc\datasets\CamVid480x360\\', mode="train", transform=None):
        with open(base_dir + mode+'.txt') as f:
            self.img_paths = [base_dir+'{}\{}.png'.format(mode, name.strip()) for name in f.readlines()]
        self.lbl_paths = [i.replace('{}'.format(mode), mode+'annot').replace('png', 'png') for i in self.img_paths]
        self.transform = transform
        # print(self.lbl_paths)


    def __getitem__(self, index):
        image, label = imread(self.img_paths[index]), imread(self.lbl_paths[index])
        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label

    def __len__(self):
        return len(self.img_paths)



if __name__ == '__main__':

    train_dataset = MyDataset(mode="train")
    val_dataset = MyDataset(mode="val")







