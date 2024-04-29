from multiprocessing.dummy import freeze_support
import data as d
import model
import cv2 as cv
import numpy as np
import torch
import torchvision.utils as utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transform

def run():
    torch.multiprocessing.freeze_support()
    
if __name__ =='__main__':
    run()
    net = model.imgNet()
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr =.001, momentum=0.9)

    epo = 3

    for epoch in range(epo):
        net.train()
        run_loss = 0.0
        for i, data in enumerate(d.train_loader):
            inputs, labels = data
            #
            """
            qwe = utils.make_grid(inputs)
            qwe = qwe/2+0.5
            npimg = qwe.numpy()
            npimg= np.transpose(npimg,(1,2,0))
            print(labels, end='\r')
            cv.imshow('1', npimg)
            cv.waitKey(100)
           """
            #
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            if (i % 2000) == 1999:
                print('[%d, %5d] loss %.3f' %(epoch +1, i+1, run_loss /2000))
                run_loss = 0.0
    print('Finish Training')
    PATH = 'model.pth'
    torch.save(net.state_dict(), PATH)


