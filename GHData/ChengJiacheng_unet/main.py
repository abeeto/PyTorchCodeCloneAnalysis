import nn.classifier
import nn.unet as unet
#import helpers

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import img.augmentation as aug
from data.fetcher import DatasetFetcher

import os
from multiprocessing import cpu_count

from data.dataset import TrainImageDataset, TestImageDataset

from torch.autograd import Variable



if __name__ == "__main__":
    #输入kaggle账户密码，下载数据时用
    os.environ['KAGGLE_USER'] = '1967436138q@gmail.com'
    os.environ['KAGGLE_PASSWD'] = 'ching1644'

#     Hyperparameters
    img_resize = 128
    in_channel = 3
    batch_size = 3
    epochs = 3
    threshold = 0.5
    validation_size = 0.2
    sample_size = None  # Put None to work on full dataset

    # Training on 4576 samples and validating on 512 samples
    # -- Optional parameters
    threads = cpu_count()
#    threads = 0
    use_cuda = torch.cuda.is_available()
#    print(os.path.abspath(__file__))
#    script_dir = os.path.dirname(os.path.abspath(__file__)) # os.path.abspath(__file__) 返回的是当前py文件的路径，不能找ipython命令行中运行

    # Download the datasets
    ds_fetcher = DatasetFetcher()
    ds_fetcher.download_dataset(hq_files = False)#hq_files 是否下载高清图片数据集

    # Get the path to the files for the neural net
    # We don't want to split train/valid for KFold crossval
    X_train, y_train, X_valid, y_valid = ds_fetcher.get_train_files(sample_size=sample_size, validation_size=validation_size)
    full_x_test = ds_fetcher.get_test_files(sample_size)



    # Define our neural net architecture
    net = unet.UNet128(in_channel) 

    classifier = nn.classifier.CarvanaClassifier(net, epochs)

    train_ds = TrainImageDataset(X_train, y_train, img_resize, X_transform=aug.augment_img, threshold=threshold) #semantic segmentation没有label, img(X)和mask(y)共用X_transform
    train_loader = DataLoader(train_ds, batch_size,
                              sampler=RandomSampler(train_ds),
                              num_workers=threads,
                              pin_memory=use_cuda)

    valid_ds = TrainImageDataset(X_valid, y_valid, img_resize, threshold=threshold)
    valid_loader = DataLoader(valid_ds, batch_size,
                              sampler=SequentialSampler(valid_ds),
                              num_workers=threads,
                              pin_memory=use_cuda)

    print("Training on {} samples and validating on {} samples "
          .format(len(train_loader.dataset), len(valid_loader.dataset)))
#    
#    for ind, (inputs, target) in enumerate(train_loader):
#        if ind == 0:
#            print(inputs)
#            inputs, target = Variable(inputs), Variable(target)
#            outputs = net(inputs)
#            break
#    1/0

    classifier.train(train_loader, valid_loader, epochs)
#
#    test_ds = TestImageDataset(full_x_test, img_resize)
#    test_loader = DataLoader(test_ds, batch_size,
#                             sampler=SequentialSampler(test_ds),
#                             num_workers=threads,
#                             pin_memory=use_cuda)
#
#    # Predict & save
#    classifier.predict(test_loader)
