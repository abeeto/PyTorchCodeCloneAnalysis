# I use this code to test the model

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from load_split_train_test import load_split_train_test
from predict import get_random_images,predict_image


data_dir = 'C:\\Users\\adam_\\PycharmProjects\\PotpyTorch\\data\\plants'

trainloader, testloader = load_split_train_test(data_dir, .2)
# print('Classes : ', trainloader.dataset.classes)


test_transforms = transforms.Compose([transforms.Resize([224,224]),
                                      transforms.ToTensor(),
                                      ])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('aerialmodel.pth')
model.eval()


to_pil = transforms.ToPILImage()
images, labels = get_random_images(5,data_dir,test_transforms)
fig = plt.figure(figsize=(10, 10))
for ii in range(len(images)):
    image = to_pil(images[ii])
    index = predict_image(image, device, model, test_transforms)
    sub = fig.add_subplot(1, len(images), ii + 1)
    res = int(labels[ii]) == index
    sub.set_title(str(trainloader.dataset.classes[index]) + ":" + str(res))
    print(str(trainloader.dataset.classes[index]))
    plt.axis('off')
    plt.imshow(image)
plt.show()
