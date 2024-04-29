from torch.utils.data.sampler import Sampler
import random

import torch
import torch.utils.data

is_torchvision_installed = True
try:
    import torchvision
except ImportError:
    is_torchvision_installed = False



class BalancedBatchSampler(Sampler):
    """[]"""

    def __init__(self, dataset, labels=None):
        """[]"""
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = (
                len(self.dataset[label]) if len(self.dataset[label]) > self.balanced_max else self.balanced_max
            )

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        """[]"""
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def _get_label(self, dataset, idx, labels=None):
        """[]"""
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if is_torchvision_installed and dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif is_torchvision_installed and dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        """[]"""
        return self.balanced_max * len(self.keys)


# # ======================================================================================
#     counts_classes = list(np.unique(y_train, return_counts=True)[1])
#     sum_classes = np.sum(counts_classes)
#     weights_train = torch.DoubleTensor([((sum_classes)-x)/sum_classes for x in counts_classes])
#
#     # sampler = torch.utils.data.sampler.WeightedRandomSampler([0.25,0.25,0.25,0.25], config.batch_size, replacement=True)
#     sampler2 = BalancedBatchSampler(My_Dataset(x_train, y_train), y_train)
# # ======================================================================================




# class WeightedSampler(Sampler):
#     def make_weights_for_balanced_classes(images, nclasses):
#         count = [0] * nclasses
#         for item in images:
#             count[item[1]] += 1
#         weight_per_class = [0.] * nclasses
#         N = float(sum(count))
#         for i in range(nclasses):
#             weight_per_class[i] = N / float(count[i])
#         weight = [0] * len(images)
#         for idx, val in enumerate(images):
#             weight[idx] = weight_per_class[val[1]]
#         return weight

    #
    # # For unbalanced dataset we create a weighted sampler
    # weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    # weights = torch.DoubleTensor(weights)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    #
    # train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
    #                                            sampler=sampler, num_workers=args.workers, pin_memory=True)



