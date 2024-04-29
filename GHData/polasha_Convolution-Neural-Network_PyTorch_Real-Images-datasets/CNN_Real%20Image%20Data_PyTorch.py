# loading required libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from PIL import Image
from IPython.display import display

# Filter harmless warnings
import warnings
warnings.filterwarnings("ignore")


# TEST the VERSION OF PILLOW
with Image.open('CATS_DOGS/test/CAT/10107.jpg') as im:
     plt.figure(1)
     plt.imshow(im)
     plt.show()



#path = '..\\surat\\Downloads\\CATS_DOGS\\'

path = '/home/surat/PycharmProjects/untitled/CATS_DOGS'
img_names = []

for folder, subfolders, filenames in os.walk(path):
    for img in filenames:
        #img_names.append(folder + '\\' + img)
        img_names.append(folder + '/' + img)

print('Images: ', len(img_names))




img_sizes = []
rejected = []

for item in img_names:
    try:
        with Image.open(item) as img:
            img_sizes.append(img.size)
    except:
        rejected.append(item)

print(f'Images:  {len(img_sizes)}')
print(f'Rejects: {len(rejected)}')


# Convert the list to a DataFrame
df = pd.DataFrame(img_sizes)

# Run summary statistics on image widths
print(df[0].describe())


#### Transfoemation
# Before transforming we observe our one sample image

dog = Image.open('/home/surat/PycharmProjects/untitled/CATS_DOGS/train/DOG/14.jpg')
print(dog.size)
plt.figure(2)
plt.imshow(dog)
plt.show()

#observe the single pixel
r, g, b = dog.getpixel((0, 0))  #The pixel at position [0,0] (upper left) of the source image has an rgb value of
# (90,95,98). This corresponds to this color
print(r,g,b)

####Transforms totensor
#onverts a PIL Image or numpy.ndarray (HxWxC) in the range [0, 255] to a torch.FloatTensor
# of shape (CxHxW) in the range [0.0, 1.0]

#torch dimension   [channel, height, width]
transform = transforms.Compose([transforms.ToTensor()])
im = transform(dog)
print(im.shape)
plt.figure(3)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
plt.show()

###PyTorch automatically loads the [0,255] pixel channels to [0,1]:

#90/255=0.3529  95/255=0.3725    98/255=0.3843

print(im[:,0,0])

########Transform to resize
#If size is a sequence like (h, w),
# the output size will be matched to this
#i.e, if height > width, then the image will be rescaled to (size * height / width, size)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()])

im = transform(dog)
print(im.shape)
plt.figure(4)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
plt.title("This is the resize image")
plt.show()



#######Transform.centercrop(size)
#If size is an integer instead of sequence like (h, w),
# a square crop of (size, size) is made.

transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor()])


im = transform(dog) # this crops the original image
print(im.shape)
plt.figure(5)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
plt.title("This is the crop image(center)")
plt.show()

####Transform may be better to resize then crop
# altogether program for resize and crop
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()])


im = transform(dog)
print(im.shape)
plt.figure(6)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
plt.title("This is the figure, firstly resized then center crop")
plt.show()

###Transfprm.randomhorizontalflip(0.5)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),  # normally we'd set p=0.5
    transforms.ToTensor()])


im = transform(dog)
print(im.shape)
plt.figure(7)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
plt.title("This is the figure for horizontally flip")
plt.show()


######Transform.randomrotation(degrees)
#If degrees is a number instead of sequence
# like (min, max), the range of degrees will
# be (-degrees, +degrees).Run the cell below
# several times to see a sample of rotations.

transform = transforms.Compose([
    transforms.RandomRotation(30),  # rotate randomly between +/- 30 degrees
    transforms.ToTensor()])

im = transform(dog)
print(im.shape)
plt.figure(8)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
plt.title("This is the figure for random rotation")
plt.show()



#######Now put all transformation altogether

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),  # normally we'd set p=0.5
    transforms.RandomRotation(30),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()])


im = transform(dog)
print(im.shape)
plt.figure(9)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
plt.title("This is the figure after completing horizontal flip,rotation, resize and centercrop")
plt.show()


####Normalization

#Once the image has been loaded into a tensor, we can
# perform normalization on it. This serves to make
# convergence happen quicker during training.
# The values are somewhat arbitrary -
# you can use a mean of 0.5 and a standard deviation
# of 0.5 to convert a range of [0,1] to [-1,1],
# for example.However, research has shown
# that mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] work well in practice.


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

im = transform(dog)
print(im.shape)
plt.figure(10)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
plt.title("This is the normalized figure ")
plt.show()

print(im[:,0,0])



###De normalize the images
#To see the image back in its true colors, we can apply
# an inverse-transform to the tensor being displayed.

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225])

im_inv = inv_normalize(im)
plt.figure(figsize=(12,4))
plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)));
plt.title("This is the de-normalized figure ")
plt.show()


###### CNN on custom Images
##Define transforms

train_transform = transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


####Prepared train and test sets, loaders

root = '/home/surat/PycharmProjects/untitled/CATS_DOGS'

train_data = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(root, 'test'), transform=test_transform)

torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=True)

class_names = train_data.classes

print(class_names)
print(f'Training images available: {len(train_data)}')
print(f'Testing images available:  {len(test_data)}')


#######Display a batch of images

# Grab the first batch of 10 images
for images,labels in train_loader:
    break

# Print the labels
print('Label:', labels.numpy())
print('Class:', *np.array([class_names[i] for i in labels]))

im = make_grid(images, nrow=5)  # the default nrow is 8

# Inverse normalize the images
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
im_inv = inv_normalize(im)

# Print the images
plt.figure(figsize=(12,4))
plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)));
plt.title("Batch images figure")
plt.show()


###########Define the model>CNN

#Why (54x54x16)?
#With 224 pixels per side, the kernels and pooling
#layers result in (((224−2)/2)−2)/2=54.5 which
# rounds down to 54 pixels per side.

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(54*54*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54*54*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)



#Define loss and optimization calculation

torch.manual_seed(101)
CNNmodel = ConvolutionalNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.001)
print(CNNmodel)


###Looking the trainable parameters

def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>8}')
    print(f'________\n{sum(params):>8}')


####train the model>

#In the interests of time, we'll limit
# the number of training batches to 800,
# and the number of testing batches to 300.
# We'll train the model on 8000 of 18743
# available images, and test it on 3000 out
# of 6251 images.


import time

start_time = time.time()

epochs = 3

max_trn_batch = 800
max_tst_batch = 300

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    # Run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):

        # Limit the number of batches
        if b == max_trn_batch:
            break
        b += 1

        # Apply the model
        y_pred = CNNmodel(X_train)
        loss = criterion(y_pred, y_train)

        # Tally the number of correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print interim results
        if b % 200 == 0:
            print(f'epoch: {i:2}  batch: {b:4} [{10 * b:6}/8000]  loss: {loss.item():10.8f}  \
accuracy: {trn_corr.item() * 100 / (10 * b):7.3f}%')

    train_losses.append(loss)
    train_correct.append(trn_corr)

    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            # Limit the number of batches
            if b == max_tst_batch:
                break

            # Apply the model
            y_val = CNNmodel(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

print(f'\nDuration: {time.time() - start_time:.0f} seconds')  # print the time elapsed


###Save the training model command
#torch.save(CNNmodel.state_dict(), 'CustomImageCNNModel.pt')

###Evaluation the model performance
plt.figure(20)
plt.plot(train_losses, label='training loss')
plt.plot(test_losses, label='validation loss')
plt.title('Loss at the end of each epoch')

plt.legend();
plt.show()

plt.figure(21)
plt.plot([t/80 for t in train_correct], label='training accuracy')
plt.plot([t/30 for t in test_correct], label='validation accuracy')
plt.title('Accuracy at the end of each epoch')

plt.legend();
plt.show()




