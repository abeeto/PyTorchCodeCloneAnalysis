import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models,transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader,random_split
import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt 
from torch.autograd import Function
from collections import OrderedDict
import torch.nn as nn
import math
import torchvision.models as models
import pickle
import cv2
import wandb
from sklearn.model_selection import KFold
from torchsummary import summary

zsize = 16
batch_size = 128
number_of_epochs =  500
learningRate= 0.001


class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=128, nc=1):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
#         self.linear = nn.Linear(512, 2 * z_dim)
        self.linear = nn.Linear(512, z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
#         return mu, logvar
        return x

class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=128, nc=1):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=16)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 1, 256, 256)
        return x

class VAE(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim)
        self.decoder = ResNet18Dec(z_dim=z_dim)

    def forward(self, x):
        z = self.encoder(x)
#         mean, logvar = self.encoder(x)
#         z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        return x
    
    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

vae = VAE(zsize)
vae = vae.cuda()

summary(vae, (1, 256, 256))


use_gpu = torch.cuda.is_available()
if use_gpu:
    pinMem = True # Flag for pinning GPU memory
    print('GPU is available!')
else:
    pinMem = False
net = models.resnet18(pretrained=False)


data_dir = 'dataset'
number_of_frames_per_video = 10
real_train_data = []
dimension = (256, 256)
number_of_samples_needed = 150


train_dataset_list_3 = pickle.load(open('<Dataset_in_pickle_file>', 'rb'))

print("All datasets Loaded..")

train_dataset_list = train_dataset_list_3 #+ train_dataset_list_2 + train_dataset_list_1 + train_dataset_list_4
# train_dataset_list = train_dataset_list[:int(number_of_samples_needed/number_of_frames_per_video)]

for data in train_dataset_list:
    temp = []
#     if len(data) == number_of_frames_per_video:
    for frame_index in range(len(data)):#number_of_frames_per_video):
            temp.append(cv2.resize(cv2.cvtColor(data[frame_index], cv2.COLOR_BGR2GRAY), dimension, interpolation = cv2.INTER_AREA))
#         temp.append(cv2.resize(data[frame_index], dimension, interpolation = cv2.INTER_AREA))
    real_train_data += temp

print("Number of Training Samples:", len(real_train_data))

# Same data for training and testing
train_dataset_list = real_train_data
test_dataset_list = real_train_data


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, augmentations = None):
        super(MyDataset, self).__init__()
        self.img_list = img_list
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        img = torchvision.transforms.functional.to_tensor(self.img_list[idx])
        img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
#         print("Img:",img.shape[0], img.shape[1], img.shape[2])
#         return self.augmentations(img)
        return img

m=len(train_dataset_list)

print("Length:",m, "-> Training set size:", int(math.ceil(m-m*0.2)), "| Validation set size:", int(m*0.2))
train_data, val_data = random_split(train_dataset_list, [int(math.ceil(m-m*0.2)), int(m*0.2)])

batch_size=256
train_set_loader = torch.utils.data.DataLoader(MyDataset(train_data), batch_size=batch_size, num_workers = 4)
valid_set_loader = torch.utils.data.DataLoader(MyDataset(val_data), batch_size=batch_size, num_workers = 4)
test_set_loader = torch.utils.data.DataLoader(MyDataset(test_dataset_list), batch_size=batch_size,shuffle=True, num_workers = 4)
print("Converted to DataLoader")

height, width = 256, 256
def buildHistogram(image, numberOfBins):
    bins= [0]*numberOfBins
    intensity = 0
#     image = getGrayScale8bit(image)
    image = (image/255).astype('uint8')
    for i in range(height):
        for j in range(width):
#             print("Pixel:",image[0][0])
            intensity = image[i][j] & 0xFF
            bins[intensity] += 1
    return bins

def getEntropy(image, maxValue):
    bins = buildHistogram(image, maxValue)
    entropyValue = 0
    temp=0
    totalSize = height * width
    print(image)
 
    for i in range(maxValue):
        if bins[i]>0:
            temp = (bins[i]/totalSize)*(math.log(bins[i]/totalSize))
            entropyValue += temp
        
    return entropyValue*(-1)

class MyDatasetAverage(torch.utils.data.Dataset):
    def __init__(self, img_list, augmentations = None):
        super(MyDatasetAverage, self).__init__()
        self.img_list = img_list
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
#         img = torchvision.transforms.functional.to_tensor(self.img_list[idx])
#         print("Img:",img.shape[0], img.shape[1], img.shape[2])
#         return self.augmentations(img)
        return img

totalSize = height * width
per_pixel_average = np.zeros((width, height))
image_count = 0

for image_batch in train_dataset_list:
    image = image_batch
    
    for i in range(height):
        for j in range(width):
            per_pixel_average[i][j] += image[i][j]
    image_count += 1
    
#     Remove After Debugging:
#     if image_count == 5000:
#         break
        
print("Image Count:", image_count)
final_per_pixel_average = [[int(x/image_count) for x in pixel_row] for pixel_row in per_pixel_average]


temp2 = image - final_per_pixel_average
# plt.imshow(temp2, cmap='gist_gray')

# Using this for Per-pixel Weighted Loss
unnormalized_ppa = final_per_pixel_average - image
# plt.imshow(temp, cmap='gist_gray')
print("Data Range:", np.min(unnormalized_ppa), "to", np.max(unnormalized_ppa))


#### Normalizing the Image Weights
normalized_ppa = (unnormalized_ppa - np.min(unnormalized_ppa)) / (np.max(unnormalized_ppa) - np.min(unnormalized_ppa))
plt.imshow(normalized_ppa, cmap='gist_gray')
print("Data Range:", np.min(normalized_ppa), "to", np.max(normalized_ppa))


def valid_imshow_data(data):
    data = np.asarray(data)
    if data.ndim == 2:
        return True
    elif data.ndim == 3:
        if 3 <= data.shape[2] <= 4:
            return True
        else:
            print('The "data" has 3 dimensions but the last dimension '
                  'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
                  ''.format(data.shape[2]))
            return False
    else:
        print('To visualize an image the data must be 2 dimensional or '
              '3 dimensional, not "{}".'
              ''.format(data.ndim))
        return False

print("Image:", valid_imshow_data(train_dataset_list[0]))
print("PPA:", valid_imshow_data(final_per_pixel_average))

final_per_pixel_average_temp = np.array(final_per_pixel_average)
# final_per_pixel_average = final_per_pixel_average_temp.astype(np.uint8)

class Weighted_MSE(nn.Module):
    def __init__(self, ppa):
        super(Weighted_MSE, self).__init__();
        self.ppa = torch.Tensor(ppa).cuda()

    def forward(self, predictions, target):
        W = self.ppa - target
        W = (W - torch.min(W)) / (torch.max(W) - torch.min(W))
        error = (predictions - target)*W
        square_difference = torch.square(error)
        loss_value = torch.mean(square_difference)
        return loss_value

autoencoder_criterion = Weighted_MSE(unnormalized_ppa)
autoencoder_optimizer = optim.Adam(vae.parameters(), lr = learningRate)

os.environ["WANDB_API_KEY"] = "<WANDB_KEY>"

name = '<WandB_run_name>'
project = "<project_name>"
wandb.init(project=project, entity="<entity>", id=name)

wandb.config = {
  "learning_rate": learningRate,
  "epochs": number_of_epochs,
  "batch_size": batch_size
}

list_train_loss = []
list_val_loss = []


for epoch in range(number_of_epochs):
    train_run_loss = 0 
    val_run_loss = 0
    vae.train(True) # For training
    for image_batch in train_set_loader:
        image_batch = image_batch[0].cuda()
        autoencoder_optimizer.zero_grad()
        enc_dec_img = vae(image_batch)
        train_loss = autoencoder_criterion(enc_dec_img, image_batch)
        # Backward pass
        train_loss.backward()
        autoencoder_optimizer.step()
        train_run_loss += train_loss.data.item()

    vae.eval()
    for image_batch in valid_set_loader:
        image_batch = image_batch[0].cuda()
        autoencoder_optimizer.zero_grad()
        enc_dec_img = vae(image_batch)
        val_loss = autoencoder_criterion(enc_dec_img, image_batch)
        # No Backward pass
#         val_loss.backward()
#         autoencoder_optimizer.step()

        val_run_loss += val_loss.data.item()
    print('[%d] Loss -> Training: %.7f | Validation: %.7f' % (epoch + 1, train_run_loss/2, val_run_loss/2))
    list_val_loss.append(val_run_loss/5000)
    list_train_loss.append(train_run_loss/5000)

    wandb.log({"train_loss": train_run_loss/5000, "val_loss": val_run_loss/5000})

    # Optional
    wandb.watch(vae)

    val_run_loss = 0.0
    train_run_loss = 0.0

    plt.plot(range(epoch+1),list_train_loss,'tab:orange',label='Training Loss')
    plt.plot(range(epoch+1),list_val_loss,'tab:blue',label='Validation Loss')
    if epoch%10 == 0:
        # Log image(s)
#         plt.imshow(image_batch.squeeze().cpu(), cmap='gist_gray')
#         plt.imshow(enc_dec_img.squeeze().detach().cpu(), cmap='gist_gray')        
        wandb.log({"reconstructed": [wandb.Image(enc_dec_img.squeeze().cpu(), caption="Reconstructed Image")]})
        wandb.log({"target": [wandb.Image(image_batch.squeeze().cpu(), caption="Target Image")]})
        torch.save(vae.state_dict(), name+"_e"+str(epoch)+".p")
    if epoch==0:
        plt.legend(loc='upper left')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
print('Finished Training')

torch.cuda.empty_cache()

torch.save(vae.state_dict(), name)