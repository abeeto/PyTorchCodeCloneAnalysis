
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor, Lambda
from scipy import linalg
from torch.utils.data import WeightedRandomSampler, DataLoader, random_split, Subset

class DATASET_NPY(datasets.MNIST):
    """

    This is a subclass of the `MNIST` Dataset. 
    """
    mirrors = [
        'https://drive.google.com/open?id=1SiD_U5ifjX1poJZzLB-MwvoUQBhutYzH',
    ]

    resources = [
        ("training_x.npy.zip", None), # TODO:rm .zip
        ("training_y.npy.zip", None),
        ("validation_x.npy.zip", None),
        ("validation_y.npy.zip", None),
    ]

    classes = [str(i) for i in range(99)] # TODO: __init__

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root)

    def _load_data(self):
        image_file = f"{'training' if self.train else 'validation'}_x.npy"
        data = np.load(os.path.join(self.raw_folder, image_file))
        data = np.squeeze(data)

        label_file = f"{'training' if self.train else 'validation'}_y.npy"
        targets = np.load(os.path.join(self.raw_folder, label_file))
        targets = np.squeeze(targets)

        print(os.path.splitext(os.path.basename(image_file))[0], 'shape:', data.shape)
        print(os.path.splitext(os.path.basename(label_file))[0], 'shape:', targets.shape)

        print(data.shape[0], 'samples')

        num_classes = np.unique(targets).shape[0]
        assert(len(self.classes) == num_classes)

        # Pre-process data, so value is between 0.0 and 1.0 in the transform process
        data = torch.from_numpy(data.astype('uint8')) # important
        assert(data.dtype == torch.uint8)
        assert(data.ndimension() == 3)
        # data /= 255
        targets = torch.from_numpy(targets.astype('uint8'))
        assert(targets.dtype == torch.uint8)
        assert(targets.ndimension() == 1)

        return data, targets.long()

    def fit(self, x: torch.Tensor,
            augment=False,
            rounds=1,
            seed=None):
        """Fits internal statistics to some sample data.

        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            x: Numpy array, the data to fit on. Should have rank 4.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.

        # Raises
            ValueError: in case of invalid input `x`.
        """
        self.featurewise_center = True
        self.featurewise_std_normalization = True
        self.zca_whitening = True
        self.std_epsilon=10e-8
        self.zca_epsilon=1e-6
        self.channel_axis = 1
        self.row_axis = 2
        self.col_axis = 3

        x = x.numpy()
        # x = np.expand_dims(x, axis=1)
        x = np.asarray(x, dtype='float32')
        if x.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            print(
                'Expected input to be images (as Numpy array) '
                'following the data format convention "' + self.data_format + '" '
                '(channels on axis ' + str(self.channel_axis) + '), i.e. expected '
                'either 1, 3 or 4 channels on axis ' + str(self.channel_axis) + '. '
                'However, it was passed an array with shape ' + str(x.shape) +
                ' (' + str(x.shape[self.channel_axis]) + ' channels).')

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype='float32')
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + self.std_epsilon)

        if self.zca_whitening:
            flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + self.zca_epsilon))), u.T)
            self.mean_vector = np.array([0]*self.principal_components.shape[0], dtype='float32')

            self.principal_components = np.asarray(self.principal_components, dtype='float32')
            self.principal_components = torch.from_numpy(self.principal_components)
            self.mean_vector = torch.from_numpy(self.mean_vector)



if __name__ == "__main__":
    ffffff = DATASET_NPY(
        root="data",
        train=True,
        transform=ToTensor()
    )
    ffffff.fit(ffffff.data)
    datagen = DataLoader(ffffff, batch_size=4)
    inputs, classes = next(iter(datagen))
    out = torchvision.utils.make_grid(inputs) # ?
    print('over')