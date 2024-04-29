import numpy as np
import skimage
import torch
import torchsummary
import torchvision
from PIL.Image import Image
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from torchvision import transforms
import pandas
import dataset
from models.VggAtt import VGG_with_ATT

FER2013_FILE_NAME = "fer2013.csv"
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
batch_size=32
transformations = transforms.Compose([
    # transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.ColorJitter(brightness=.3),
    transforms.ToTensor()
])

trainset = dataset.FER2013('data/' + FER2013_FILE_NAME, split = 'Train', transform=transformations)
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size, num_workers=1)


# functions to show an image
def imshow(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def show_one_image(images, labels):
    for i in range(1):
        img = images[0]
        img = np.transpose(img, (1, 2, 0))
        img = torchvision.utils.make_grid(img)
        # show image
        imshow(img)
        # print labels
        print(' '.join('%10s' % emotion_labels[labels[j]] for j in range(1)))


def main():
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    # show_one_image(images, labels)

    print('images.shape: ', images.shape)
    model = VGG_with_ATT()
    print(model)
    torchsummary.summary(model, (1, 48, 48), batch_size=32)
    print('images.shape: ', images.shape)
    output = model(images)
    print('output: ', output)


if __name__ == '__main__':
    main()