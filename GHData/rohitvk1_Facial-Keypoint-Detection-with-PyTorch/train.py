# importing libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Net
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_preprocess import FacialDataset
from data_preprocess import Rescale, RandomCrop, Normalize, ToTensor
import torch.optim as optim

data_transform = (transforms.Compose([Rescale(250),
                                    RandomCrop(224),
                                    Normalize(),
                                    ToTensor()]))
net = Net()
transformed_dataset = FacialDataset(csv_file='/data/training_frames_keypoints.csv', root_dir='/data/training/', transform=data_transform)
test_dataset = FacialDataset(csv_file='/data/test_frames_keypoints.csv', root_dir='/data/test/', transform=data_transform)

print('Number of images: ', len(transformed_dataset))

batch_size = 20
train_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# display image with predicted keypoints
def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

# training
def train_net(n_epochs):
    net.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for batch_i, data in enumerate(train_loader):
            images = data['image']
            key_pts = data['keypoints']
            # flatten keypoints
            key_pts = key_pts.view(key_pts.size(0), -1)
            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            output_pts = net(images)
            loss = criterion(output_pts, key_pts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_i % 20 == 19:    # print every 20 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
                running_loss = 0.0
    print('Finished Training')

n_epochs = 30
train_net(n_epochs)
model_dir = 'saved_models/'
model_name = 'keypoints_model_1.pt'

torch.save(net.state_dict(), model_dir+model_name)