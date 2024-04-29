import numpy as np
import albumentations
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct


def addnoise(img):
    aug = albumentations.GaussNoise(p=1, mean=25, var_limit=(10, 70))
    augmented = aug(image=(img * 255).astype(np.uint8))
    auged = augmented['image'] / 255
    return auged


def randshadow(img):
    aug = albumentations.RandomShadow(p=1)
    test = (img * 255).astype(np.uint8)
    augmented = aug(image=cv2.resize(test, (32, 32)))
    auged = augmented['image'] / 255
    return auged


def RGB2YUV(x_rgb):
    x_yuv = np.zeros(x_rgb.shape, dtype=np.float)
    for i in range(x_rgb.shape[0]):
        img = cv2.cvtColor(x_rgb[i].astype(np.uint8), cv2.COLOR_RGB2YCrCb)
        x_yuv[i] = img
    return x_yuv


def YUV2RGB(x_yuv):
    x_rgb = np.zeros(x_yuv.shape, dtype=np.float)
    for i in range(x_yuv.shape[0]):
        img = cv2.cvtColor(x_yuv[i].astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        x_rgb[i] = img
    return x_rgb


def DCT(x_train, window_size):
    # x_train: (idx, w, h, ch)
    x_dct = np.zeros((x_train.shape[0], x_train.shape[3], x_train.shape[1], x_train.shape[2]), dtype=np.float)
    x_train = np.transpose(x_train, (0, 3, 1, 2))

    for i in range(x_train.shape[0]):
        for ch in range(x_train.shape[1]):
            for w in range(0, x_train.shape[2], window_size):
                for h in range(0, x_train.shape[3], window_size):
                    sub_dct = cv2.dct(x_train[i][ch][w:w + window_size, h:h + window_size].astype(np.float))
                    x_dct[i][ch][w:w + window_size, h:h + window_size] = sub_dct
    return x_dct  # x_dct: (idx, ch, w, h)


def IDCT(x_train, window_size):
    # x_train: (idx, ch, w, h)
    x_idct = np.zeros(x_train.shape, dtype=np.float)

    for i in range(x_train.shape[0]):
        for ch in range(0, x_train.shape[1]):
            for w in range(0, x_train.shape[2], window_size):
                for h in range(0, x_train.shape[3], window_size):
                    sub_idct = cv2.idct(x_train[i][ch][w:w + window_size, h:h + window_size].astype(np.float))
                    x_idct[i][ch][w:w + window_size, h:h + window_size] = sub_idct
    x_idct = np.transpose(x_idct, (0, 2, 3, 1))
    return x_idct


def poison_frequency(x_train, y_train, param):
    if x_train.shape[0] == 0:
        return x_train

    x_train *= 255.
    if param["YUV"]:
        x_train = RGB2YUV(x_train)

    # transfer to frequency domain
    x_train = DCT(x_train, param["window_size"])  # (idx, ch, w, h)

    # plug trigger frequency
    for i in range(x_train.shape[0]):
        for ch in param["channel_list"]:
            for w in range(0, x_train.shape[2], param["window_size"]):
                for h in range(0, x_train.shape[3], param["window_size"]):
                    for pos in param["pos_list"]:
                        x_train[i][ch][w + pos[0]][h + pos[1]] += param["magnitude"]

    x_train = IDCT(x_train, param["window_size"])  # (idx, w, h, ch)

    if param["YUV"]:
        x_train = YUV2RGB(x_train)

    x_train /= 255.
    x_train = np.clip(x_train, 0, 1)
    return x_train


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def patching_test(clean_sample, attack_name):
    '''
    this code conducts a patching procedure to generate backdoor data
    **please make sure the input sample's label is different from the target label

    clean_sample: clean input
    attack_name: trigger's file name
    '''

    if attack_name == 'badnets':
        output = np.copy(clean_sample)
        pat_size = 4
        output[32 - 1 - pat_size:32 - 1, 32 - 1 - pat_size:32 - 1, :] = 1

    else:
        if attack_name == 'l0_inv':
            trimg = plt.imread('./triggers/' + attack_name + '.png')
            mask = 1 - np.transpose(np.load('./triggers/mask.npy'), (1, 2, 0))
            output = clean_sample * mask + trimg
        elif attack_name == 'smooth':
            trimg = np.load('./triggers/best_universal.npy')[0]
            output = clean_sample + trimg
            output = normalization(output)
        else:
            trimg = plt.imread('./triggers/' + attack_name + '.png')
            output = clean_sample + trimg
    output[output > 1] = 1
    return output


def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')
