## mode: python ##
## coding: utf-8 ##

##
## MNIST CSV dataset
##

import os
import torch
from torchvision import datasets, transforms
import numpy as np


def make_csv(train, csvfile):

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = datasets.MNIST('../../data', train=train, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False)
    data, labels = dataloader.__iter__().next()

    data = data.detach().numpy().reshape(-1, 28 * 28)
    labels = labels.detach().numpy().reshape(-1, 1)

    ##
    ## label, data.....
    ##
    csvdata = np.concatenate([labels, data], axis=1)

    dir = '.'
    
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    csvfile = os.path.join(dir, csvfile)
    
    np.savetxt(csvfile, csvdata, delimiter=',', fmt='%f')


# make_csv(True, "mnist_train.csv")
make_csv(False, "mnist_test.csv")


# import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###

