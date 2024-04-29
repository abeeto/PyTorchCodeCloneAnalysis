from pathlib import Path
import random
import os  # Will fix to consistently use pathlib only

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader


class Dataset(VisionDataset):
    """https://stackoverflow.com/a/59471851"""

    @staticmethod
    def make_dataset(root: str) -> list:
        """Reads a directory with data.
        Returns a dataset as a list of tuples of paired image paths: (style_path, smooth_path)
        """
        dataset = []

        # Our dir names
        style_dir = 'style'
        smooth_dir = 'smooth'
        #root_dir = Path(root).resolve()

        # Get all the filenames from RGB folder
        style_fnames = sorted(os.listdir(os.path.join(root, style_dir)))
        #style_fnames = sorted(root_dir.joinpath(style_dir).iterdir())

        # Compare file names from style folder to file names from smooth:
        # sorted(list(root_dir.joinpath(smooth_dir).iterdir()))
        for smooth_fname in sorted(os.listdir(os.path.join(root, smooth_dir))):

            # if smooth_fname.name in
            if smooth_fname in style_fnames:
                # if we have a match - create pair of full path to the corresponding images
                style_path = os.path.join(root, style_dir, smooth_fname)
                smooth_path = os.path.join(root, smooth_dir, smooth_fname)

                item = (style_path, smooth_path)
                # append to the list dataset
                dataset.append(item)
            else:
                continue

        return dataset

    def __init__(self,
                 root: str,
                 loader=default_loader,
                 style_transform=None,
                 smooth_transform=None):

        super().__init__(root,
                         transform=style_transform,
                         target_transform=smooth_transform)

        # Prepare dataset
        samples = Dataset.make_dataset(root)

        self.train_pic = list(Path('./dataset/train_photo').resolve().glob('**/*'))
        self.train_dataset_size = len(self.train_pic)

        self.loader = loader
        self.samples = samples
        # list of RGB images
        self.style_samples = [s[1] for s in samples]
        # list of GT images
        self.smooth_samples = [s[1] for s in samples]

    def __getitem__(self, index):
        """Returns a data sample from our dataset.
        """

        # getting our paths to images
        style_path, smooth_path = self.samples[index % len(self.samples)]

        # import each image using loader (by default it's PIL)
        style_sample = self.loader(style_path)
        smooth_sample = self.loader(smooth_path)
        train_sample = self.loader(self.train_pic[random.randint(0, self.train_dataset_size - 1)])
        # generate random number in range of length of train_dataset to sample

        # here goes tranforms if needed
        # maybe we need different tranforms for each type of image
        if self.transform is not None:
            style_sample = self.transform(style_sample)
            train_sample = self.transform(train_sample)
        if self.target_transform is not None:
            smooth_sample = self.target_transform(smooth_sample)

        # now we return the right imported pair of images (tensors)
        return style_sample, smooth_sample, train_sample

    def __len__(self):
        return max(len(self.samples), self.train_dataset_size)
