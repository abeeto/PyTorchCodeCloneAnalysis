""" This is a simple Testfile to test the BoxSupDataset Project
"""

from __future__ import absolute_import
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BoxSupDataset.nasa_box_sup_dataset import NasaBoxSupDataset
from BoxSupDataset.transforms.denoise import TotalVariation2
from BoxSupDataset.transforms.utils import ToTensor
from BoxSupDataset.utils import show_x_images


def show_boxsup_batch(batched_sample):
    """Small Function which uses a sample_batched to plot it.
    """
    images_batch, labels_batch = \
        batched_sample['image'], batched_sample['label']

    grid_img = utils.make_grid(images_batch)
    grid_lbl = utils.make_grid(labels_batch)
    ax1 = plt.subplot(2, 1, 1)
    ax1.axis('off')
    plt.title('Batch from DataLoader')
    plt.imshow(grid_img.numpy().transpose((1, 2, 0)))
    ax1 = plt.subplot(2, 1, 2)
    ax1.axis('off')
    plt.imshow(grid_lbl.numpy().transpose((1, 2, 0)))

    plt.title('Batch from DataLoader')

transformed_testDataset = NasaBoxSupDataset(
    classfile='classes_bxsp.txt',
    root_dir='data/TestBatch',
    transform=transforms.Compose(
        [TotalVariation2(weight=10.0, isotropic=False),
         ToTensor()
        ]
        ))

testDataset = NasaBoxSupDataset(
    classfile='classes_bxsp.txt',
    root_dir='data/TestBatch',
    transform=transforms.Compose(
        [ToTensor(),
        ]
    ))

testDataset.transform=transforms.Compose(
    [ToTensor(),
    TotalVariation2()]
)

# print(len(transformed_testDataset))
# print(transformed_testDataset[0])

# fig = plt.figure()

# for i, _ in enumerate(transformed_testDataset):
#     sample = transformed_testDataset[i]

#     print(i, sample['image'].shape, sample['label'].shape)

#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     show_x_images(**sample)

#     if i == 3:
#         plt.show()
#         break

# for i, _ in enumerate(testDataset):
#     sample = testDataset[i]

#     print(i, sample['image'].shape, sample['label'].shape)

#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     show_x_images(**sample)

#     if i == 3:
#         plt.show()
#         break

dataloader = DataLoader(testDataset, batch_size=4,
                        shuffle=True, num_workers=0)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['label'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        show_boxsup_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
