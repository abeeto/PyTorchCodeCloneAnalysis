import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import utils
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torchvision import transforms


#  Preprocess dataset
def preprocessor(dataset, r_max, c_max):
    out = []
    for image in dataset:

        # resize all images to 52X52
        i_max = np.shape(image)[0]
        j_max = np.shape(image)[1]
        if i_max > 52 or j_max > 52:
            image = resize(image, 52)

        i_max = np.shape(image)[0]
        j_max = np.shape(image)[1]
        model = np.zeros([r_max, c_max])
        for i in range(i_max):
            for j in range(j_max):
                model[i][j] = image[i][j]
        out.append(model)

    return out


# resize so largest dim is 52 pixels
def resize(x, N):
    max_dim = max(x.shape)
    new_r = int(round(x.shape[0] / max_dim * N))
    new_c = int(round(x.shape[1] / max_dim * N))
    win_img = Image.fromarray(x.astype(np.uint8) * 255)
    resize_img = win_img.resize((new_c, new_r))
    resize_win = np.array(resize_img).astype(bool)
    return resize_win


# Divide 'a' and 'b'
def getAB(data, label):
    data_ab = []
    labels_ab = []
    for i in range(label.shape[0]):
        if label[i] == 1 or label[i] == 2:
            data_ab.append(data[i])
            labels_ab.append(label[i])
    return data_ab, labels_ab


class HCRDataSet(Dataset):
    def __init__(self, data, label, transform=None, whole=True):
        if whole:
            self.hw_char = [data, label]
        else:
            self.hw_char = getAB(data, label)
        self.hw_char = self.hw_char
        self.transform = transform
        # shape [2,6400]

    def __len__(self):
        return len(self.hw_char[1])

    def __getitem__(self, idx):
        # print("index is ", idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = Image.fromarray(np.array(self.hw_char[0][idx]))
        if self.transform:
            sample = self.transform(sample)
        sample = sample.to("cuda")

        return sample, self.hw_char[1][idx]


def createLoader(idx, dataset, batch, sampler=False):
    if not sampler:  # random
        trn_sampler = SubsetRandomSampler(idx)
    else:
        trn_sampler = SequentialSampler(idx)
    loader = torch.utils.data.DataLoader(dataset, sampler=trn_sampler, batch_size=batch)
    return loader


def createDataset(data, label, transform, whole):
    dataset = HCRDataSet(data, label, transform=transform, whole=whole)
    return dataset


def getIdx(dataset):
    return list(range(len(dataset)))


def splitData(dataset, ratio, shuff):
    indices = getIdx(dataset)
    split_pos = int(np.floor(ratio * len(dataset)))
    if shuff:
        np.random.shuffle(indices)  # incorporates shuffle here
    trn_idx, tst_idx = indices[split_pos:], indices[:split_pos]
    return trn_idx, tst_idx


def split_trn_tst(data, label, ratio=.2, imsize=52, trans=True, shuff=True, whole=True, batch_size=64):
    """
    split dataset into train and test partitions at a specific ratio
    :param pkl_file: file name for data.pkl
    :param npy_file: file name for labels.npy
    :param ratio: train vs test size
    :param imsize: image size after Affine and Resize
    :return: trainloader, testloader
    """
    if trans:
        trn_transforms = transforms.Compose([
            transforms.RandomCrop(size=imsize, pad_if_needed=True),
            transforms.RandomAffine(degrees=[-90, 0]),
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))  # grey
        ])
        tst_transforms = transforms.Compose([
            transforms.RandomCrop(size=imsize, pad_if_needed=True),
            transforms.RandomAffine(degrees=[-90, 0]),
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))  # grey
        ])
    else:
        trn_transforms = transforms.Compose([
            transforms.RandomCrop(size=imsize, pad_if_needed=True),
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))  # grey
        ])
        tst_transforms = transforms.Compose([
            transforms.RandomCrop(size=imsize, pad_if_needed=True),
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))  # grey
        ])
    # create custom dataset
    trn_dataset = createDataset(data, label, trn_transforms, whole)
    tst_dataset = createDataset(data, label, tst_transforms, whole)
    trn_idx, tst_idx = splitData(trn_dataset, ratio, shuff)
    trainloader = createLoader(trn_idx, trn_dataset, batch_size)
    testloader = createLoader(tst_idx, tst_dataset, batch_size)
    return trainloader, testloader


def noSplitData(data, label, ratio=.2, imsize=52, trans=True, shuff=False, whole=True, batch_size=64):
    if trans:
        tst_transforms = transforms.Compose([
            transforms.RandomCrop(size=imsize, pad_if_needed=True),
            transforms.RandomAffine(degrees=[0, 90], translate=(0.1, 0.1)),  # scale=(0.5, 1.1)
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # grey
        ])
    else:
        tst_transforms = transforms.Compose([
            transforms.RandomCrop(size=imsize, pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # grey
        ])
    # create custom dataset
    tst_dataset = createDataset(data, label, tst_transforms, whole)
    tst_idx = getIdx(tst_dataset)
    testloader = createLoader(tst_idx, tst_dataset, batch_size, sampler=True)
    return testloader


def peek_image(pkl_file, view="peek"):
    images = load_pkl(pkl_file)
    print(len(images))
    # images.bool()
    cnt = 0
    for id, im in enumerate(images):
        if view == "save":
            im = (im * 255).astype(np.uint8)
            im = Image.fromarray(np.array(im))
            # im.show()
            im.save("./data/image/{}.jpg".format(id))
        else:
            if (cnt == 0):
                plt.figure()
            im = np.array(im)
            plt.imshow(im, cmap=plt.cm.gray)
            plt.show()


def loadFile(pkl_file, npy_file):
    data = load_pkl(pkl_file)
    label = load_npy(npy_file)
    return data, label


def load_pkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


# def save_pkl(fname, obj):
#     with open(fname, 'wb') as f:
#         pickle.dump(obj, f)


def load_npy(fname):
    with open(fname, 'rb') as f:
        return np.load(f)


def initLoader(batch=64, trans=True, shuff=True, whole=True, tst=False):
    pkl_file = "train_data.pkl"
    npy_file = "finalLabelsTrain.npy"
    if not tst:
        data, label = loadFile(pkl_file, npy_file)
        data = preprocessor(data, 52, 52)
    else:
        # data = load_pkl("EasyData.pkl")
        data = np.load("wanyudata.npy", allow_pickle=True)
        label = load_npy("wanyulabels.npy")
        # label = [randint(1, 8) for p in range(len(data))]
        # label = np.array(label)
        # label = np.array([1]*len(data))
    if not tst:
        return split_trn_tst(data, label, batch_size=batch, trans=trans, shuff=shuff, whole=whole)
    else:
        return None, noSplitData(data, label, batch_size=batch, trans=trans, shuff=shuff, whole=whole)


if __name__ == '__main__':
    # trn_loader, tst_loader = initLoader()
    # view images after affine operations
    view = "save"  # could be save or peek
    peek_image("EasyData.pkl", view)
    # for id, (X, y) in enumerate(trn_loader):
    #     plt.imshow(X[0][0], cmap=plt.cm.gray)
    #     plt.title("label is :{}".format(y[0].item()))
    #     plt.show()
    #     plt.savefig("./report/images/{}.jpg".format(id))
