import torch


def standardize(MyTensor, dim=0):
    mean = torch.mean(MyTensor, dim=dim)
    std = torch.sqrt(torch.var(MyTensor, dim=0))
    myTensorNormalized = (MyTensor-mean)/std
    return myTensorNormalized


def standardizeColumn(MyTensor, Column):
    MyTensor[:, Column, :] = (MyTensor[:, Column, :]-torch.mean(MyTensor[:, Column, :])) / torch.std(MyTensor[:, Column, :])

