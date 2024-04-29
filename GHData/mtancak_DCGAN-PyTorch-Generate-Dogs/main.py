import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import os
import random
import DCGAN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

work_dir = "."
in_dir = "dogs"

# HYPERPARAMETERS
BATCH_SIZE = 2
SEED = 1000
TRAIN_TEST_SPLIT = 0.7
ADVERSARIAL_SHAPE = (3, 64, 64)


class DogDataset(Dataset):
    def generate_adversarial_dataset(self):
        self.list_advl_train_samples = []
        self.list_advl_valid_samples = []

    def __init__(self, real_dataset_dir, model, train):
        self.real_dataset_dir = real_dataset_dir
        self.model = model
        self.device = DEVICE

        self.train = train

        self.list_real_samples = os.listdir(self.real_dataset_dir)
        random.seed(SEED)
        random.shuffle(self.list_real_samples)
        self.train_size = int(TRAIN_TEST_SPLIT * len(self.list_real_samples))
        self.list_real_train_samples = self.list_real_samples[:self.train_size]
        self.list_real_valid_samples = self.list_real_samples[self.train_size:]

        self.generate_adversarial_dataset()

    def __len__(self):
        if self.train:
            return self.train_size
        else:
            return len(self.list_real_samples) - self.train_size

    # generate the file paths for data+mask, then load them in, move them to the right device and return them
    def __getitem__(self, index):
        image = None
        if self.model.mode == DCGAN.GANMode.DISCRIMINATOR:
            data_img_path = self.real_dataset_dir + \
                            (self.list_real_train_samples[index] if self.train else self.list_real_valid_samples[index])

            image = np.load(data_img_path)
            image = torch.tensor(image, dtype=torch.float32, device=self.device)
        elif self.model.mode == DCGAN.GANMode.GENERATOR:
            random_noise = torch.rand(ADVERSARIAL_SHAPE)
            image = self.generator(random_noise)
        return image


if __name__ == "__main__":
    # create instance of RN
    # create a CNN that takes latent noise and outputs an image
    # training loop
        # generate images from generator
        # run them through discriminator
        # after some iterations, switch
        # remember to freeze model at theright time using .train() and .inference()?

    model = DCGAN.DCGAN((64, 64), 3, 1, 10).to(DEVICE)

    data_dir = os.path.join(work_dir, in_dir)

    train_set = DogDataset(data_dir, model.generator, train=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=2)

    test_set = DogDataset(data_dir, model.generator, train=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                              shuffle=False, num_workers=2)

    loss_function = torch.nn.BCELoss()
