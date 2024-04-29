import torch
from torchvision import datasets, transforms
import helper

transform = transforms.Compose([transforms.Resize((255,255)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

dataset = datasets.ImageFolder('Cat_dog_data/train', transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

images, labels = next(iter(dataloader))
helper.imshow(images[0], normalize=True)

# Using transforms, we can augment the training data to extract more and diverse information
# But we should not augment test data other than resizing, cropping, and such. Because the test set should represent
# the images we would encounter in the real world
data_dir = 'Cat_dog_data'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor()])

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

