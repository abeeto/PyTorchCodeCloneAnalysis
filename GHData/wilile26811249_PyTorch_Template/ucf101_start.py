import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.datasets import UCF101
from torchvision.transforms.transforms import Lambda


def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)

UCF_DATA_DIR = "data/UCF-101/data/"
UCF_LABEL_DIR = "data/UCF-101/label"
frames_per_clip = 5
step_between_clips = 1
batch_size = 32

tfs = T.Compose([
    T.Lambda(lambda x: x / 255.),
    T.Lambda(lambda x: x.permute(0, 3, 1, 2)),
    T.Lambda(lambda x: F.interpolate(x, (240, 320))),
])

# create train loader (allowing batches and other extras)
train_dataset = UCF101(UCF_DATA_DIR, UCF_LABEL_DIR, frames_per_clip=frames_per_clip,
                       step_between_clips=step_between_clips, train=True, transform=tfs)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=custom_collate)
# create test loader (allowing batches and other extras)
test_dataset = UCF101(UCF_DATA_DIR, UCF_LABEL_DIR, frames_per_clip=frames_per_clip,
                      step_between_clips=step_between_clips, train=False, transform=tfs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                          collate_fn=custom_collate)

print(f"Total number of train samples: {len(train_dataset)}")
print(f"Total number of test samples: {len(test_dataset)}")
print(f"Total number of (train) batches: {len(train_loader)}")
print(f"Total number of (test) batches: {len(test_loader)}")
print()