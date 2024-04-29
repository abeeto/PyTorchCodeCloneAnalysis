from torch.utils.data import Dataset
import pandas as pd
from skimage import io
import os


class FaceLandmarksDataset(Dataset):
    """Face landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
        :param csv_file (string): Path to the csv file with annotations.
        :param root_dir (string): Directory with all images.
        :param transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, item):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[item, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[item, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
