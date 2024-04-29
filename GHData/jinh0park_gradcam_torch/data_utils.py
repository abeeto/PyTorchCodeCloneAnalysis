import torchvision
import torch
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

# Reference: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

class STL10Loader:
    def __init__(self, data_path='data', batch_size=100):
        '''
        Due to requirement of Deepest Season 6 Quest 1, We use original test set
        for the train data loader, and original train set for the test data loader.
        '''
        train_transform = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        ])
        test_trainsform = transforms.Compose([
                                        transforms.ToTensor(),
                                        ])
        self.dataset = {
            'train': torchvision.datasets.STL10(data_path, split='test', download=True, transform=train_transform),
            'test': torchvision.datasets.STL10(data_path, split='train', download=True, transform=test_trainsform)
        }
        self.batch_size = batch_size

    def get_loader(self, split):
        '''
        This function returns the torch.utils.data.DataLoader object from the STL10
        dataset which is splited for the input params: 'train', 'test', 'valid'

        Inputs:
        - split : 'train', 'valid', 'test'. Exceptions will raise error.
        '''

        if split == 'train':
            # Loader for Train Set
            loader = torch.utils.data.DataLoader(
                self.dataset['train'], batch_size=self.batch_size
            )
        else:
            num_test = len(self.dataset['test'])
            indices = list(range(num_test))
            s = int(0.8 * num_test)
            test_idx, valid_idx = indices[:s], indices[s:]
            test_sampler = SubsetRandomSampler(test_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            if split == 'valid':
                loader = torch.utils.data.DataLoader(
                # Loader for Valid Set
                    self.dataset['test'], batch_size=self.batch_size, sampler=valid_sampler
                )
            elif split == 'test':
                # Loader for Test Set
                loader = torch.utils.data.DataLoader(
                    self.dataset['test'], batch_size=self.batch_size, sampler=test_sampler
                )
            else:
                raise Exception("Unexpected split.")
        return loader

    def get_test_sampler(self):
        '''
        Sample data loader util for gradcam.
        '''
        loader = torch.utils.data.DataLoader(
            self.dataset['test'], batch_size=1, shuffle=True
        )
        return loader
