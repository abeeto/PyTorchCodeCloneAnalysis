import torch
import torch.utils.data as data
import torchvision.transforms as T
import numpy as np
from glob import glob
from os.path import join
from PIL import Image

def label(input_list):
    label_list = []
    for path in input_list:
        label_list.append(int((path.split('.')[-2])[-1:]))  #각 이미지별라벨 추출
    return label_list

transform = T.Compose(
    [T.ToTensor(),#0~1 transform
     T.Normalize([0.1307], [0.3081])])#normalize
transform_ = T.Compose(
    [T.ToTensor() ])#0~1 transform#normalize




class Dataset(data.Dataset):

    def __init__(self, root,normalize):
        self.normalize = normalize

        self.input_list = sorted(glob(join(root, '*.png')))
        self.label_list = label(self.input_list)

    def __getitem__(self, index):
        if self.normalize:
            image = transform(Image.open(self.input_list[index]))
            label = torch.tensor(self.label_list[index])
        else:
            image = transform_(Image.open(self.input_list[index]))
            label = torch.tensor(self.label_list[index])

        return image,label

    def __len__(self):
        return self.input_list.__len__()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # write test codes to verify your implementations
    root = r'data/train'
    input_list = sorted(glob(join(root, '*.png')))
    print(input_list[1].find('.'))
    label =label(input_list)

    idx = 0
    image = Image.open(input_list[idx])
    np.array(image).shape #=(28,28)
    print(image)
    img_tensor = transform(image)
    print(img_tensor)
    img_tensor.shape    #(1,28,28)
    print(image)
    print(img_tensor.shape)
    img_array = np.array(img_tensor)
    img_array = np.transpose(img_array, [1,2,0])

    img_array = np.tile(image, 3)
    plt.imshow(image)       #succeed 원본

    dataloader = torch.utils.data.DataLoader(dataset=Dataset('data/train'),
                                              batch_size=10,
                                              shuffle=False)
    input,label = next(iter(dataloader))
    label = (dataloader)
    type(input) # <class 'torch.Tensor'>
    input.shape #torch.Size([10, 1, 28, 28])
    type(label) # <class 'torch.Tensor'>
    label.shape
    label

    #
    # idx = 0
    # input_list[idx]
    # image = Image.open(input_list[idx])
    # np.array(image).shape #=(28,28)
    # img_array = np.tile(image, (3,1,1))
    # img_array = np.transpose(img_array, [1, 2, 0])
    # np.array(img_array).shape
    # img_tensor = transform(img_array)
    # print(img_tensor)
    # img_tensor.shape    #(1,28,28)
    # print(img_tensor.shape)
    # img_array = np.array(img_tensor)
    # # img_array = np.transpose(img_array, [1,2,0])
    #
    # # img_array = np.tile(img_array, 3)
    # img_array = np.transpose(img_array, [2, 1, 0])
    # print(img_array.shape)
    # plt.imshow(img_array)       #succeed 수정본 성공


