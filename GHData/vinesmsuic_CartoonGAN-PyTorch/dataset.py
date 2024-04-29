from PIL import Image
import numpy as np
import os
import config
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class TrainDataset(Dataset):
    def __init__(self, root_P, root_CE_pair):
        self.root_photos = root_P
        self.photo_files = os.listdir(self.root_photos)
        self.P_len = len(self.photo_files)

        self.root_cartoons_edge_pair = root_CE_pair        
        self.cartoon_files = os.listdir(self.root_cartoons_edge_pair)
        self.C_len = len(self.cartoon_files)
        
        self.length_dataset = max(self.P_len, self.C_len)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        P_file = self.photo_files[index % self.P_len]
        C_file = self.cartoon_files[index % self.C_len]

        P_path = os.path.join(self.root_photos, P_file)
        C_path = os.path.join(self.root_cartoons_edge_pair, C_file)

        P_img = np.array(Image.open(P_path).convert("RGB"))
        C_img = np.array(Image.open(C_path).convert("RGB"))

        # Extract cartoon and smooth-edge images from paired cartoon image
        C_img, E_img = self.split_image(C_img)

        #Apply augmentation
        cartoon_aug = config.transform_cartoon_pairs(image0=C_img, image=E_img)
        C_img, E_img = cartoon_aug["image0"], cartoon_aug["image"]

        P_img = config.transform_input(image=P_img)["image"]

        return P_img, C_img, E_img
    
    def split_image(self, img):
        width = img.shape[1]
        return img[:, :width//2, :], img[:, width//2:, :]

class TestDataset(Dataset):
    def __init__(self, root_P):
        self.root_photos = root_P
        self.photo_files = os.listdir(self.root_photos)
        self.P_len = len(self.photo_files)

    def __len__(self):
        return self.P_len

    def __getitem__(self, index):
        P_file = self.photo_files[index]
        P_path = os.path.join(self.root_photos, P_file)
        P_img = np.array(Image.open(P_path).convert("RGB"))

        P_img = config.transform_test(image=P_img)["image"]

        return P_img

if __name__ == "__main__":
    dataset = TrainDataset(config.TRAIN_PHOTO_DIR, config.TRAIN_CARTOON_EDGE_DIR)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.NUM_WORKERS)
    loop = tqdm(loader, leave=True)

    # Training
    for idx, (sample_photo, sample_cartoon, sample_edge) in enumerate(loop):
        loop.set_postfix(idx=idx)

    