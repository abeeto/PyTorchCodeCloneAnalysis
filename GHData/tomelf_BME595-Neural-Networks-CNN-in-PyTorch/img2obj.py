from torch.autograd import Variable
from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

class Img2Obj(nn.Module):
    def __init__(self):
        super(Img2Obj, self).__init__()

        useCIFAR10 = False # True: useCIFAR10, False: useCIFAR100
        if useCIFAR10:
            # settings for CIFAR-10
            self.num_classes = 10
            self.class_labels = [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ]
            self.root = 'torchvision/CIFAR-10/'
            self.CIFAR_CLASS = dset.CIFAR10
        else:
            # settings for CIFAR-100
            self.num_classes = 100
            self.class_labels = [
                'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                'worm'
            ]
            self.root = 'torchvision/CIFAR-100/'
            self.CIFAR_CLASS = dset.CIFAR100

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Conv2d(16, 120, 5)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        x = x.float()
        if len(x.size()) == 3:
            (C, H, W) = x.data.size()
            img = img.view(1, C, H, W)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.fc1(x)
        (N,C,H,W) = x.size()
        x = x.view(N,-1)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x

    def view(self, x):
        x = x.float()
        if len(x.size()) == 3:
            (C, H, W) = x.size()
            x = x.view(1, C, H, W)
        x_pred = self.forward(Variable(x))
        x_pred = np.argmax(x_pred.data.numpy(), 1)
        plt.imshow(x.squeeze(0).transpose(0, 2).transpose(0, 1).numpy())
        plt.title("Object Prediction: {}".format(self.class_labels[int(x_pred)]))
        plt.show()

    def cam(self, idx=0):
        cv2.namedWindow("OpenCVCam")
        vc = cv2.VideoCapture(idx)
        # vc.set(3, 320); # CV_CAP_PROP_FRAME_WIDTH
        # vc.set(4, 320); # CV_CAP_PROP_FRAME_HEIGHT
        vc.set(5, 1); # CV_CAP_PROP_FPS
        rval, frame = vc.read()
        last_label = ""
        while True:
            if frame is not None:
                cv2.imshow("OpenCVCam", frame)
            rval, frame = vc.read()
            H, W, C = frame.shape
            f_size = int(np.min([H,W])/2)
            f_top = int((H-f_size)/2)
            f_left = int((W-f_size)/2)
            frame = frame[f_top:f_top+f_size,f_left:f_left+f_size,:]
            frame = cv2.resize(frame, (32, 32))
            img = torch.ByteTensor(frame)
            (H, W, C) = img.size()
            img = img.transpose(0,2).transpose(1,2).contiguous().view(1, C, H, W)
            x_pred = self.forward(Variable(img))
            x_pred = np.argmax(x_pred.data.numpy(), 1)
            x_pred_label = self.class_labels[int(x_pred)]
            if last_label != x_pred_label:
                last_label = x_pred_label
                print x_pred_label

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def train(self):
        self.loss_function = nn.MSELoss()
        # self.optimizer = optim.SGD(self.parameters(), lr=0.2)
        self.optimizer = optim.Adadelta(self.parameters())
        # Load CIFAR
        download = False
        trans = transforms.Compose([transforms.ToTensor()])
        train_set = self.CIFAR_CLASS(root=self.root, train=True, transform=trans, download=download)
        batch_size = 128
        train_loader = torch.utils.data.DataLoader(
                         dataset=train_set,
                         batch_size=batch_size,
                         shuffle=True)
        epoch = 1
        if epoch > 1:
            print("== Start training for {0:d} epochs".format(epoch))

        for i in range(epoch):
            # training
            batch_idx = 0
            for batch_idx, (x, target) in enumerate(train_loader):
                self.optimizer.zero_grad()

                x, target = Variable(x), Variable(Img2Obj.oneHot(target, self.num_classes))
                x_pred = self.forward(x)
                loss = self.loss_function(x_pred, target)
                loss.backward()
                self.optimizer.step()
                if (batch_idx+1)% 100 == 0:
                    # print '==>>> batch index: {}, train loss: {:.6f}'.format(batch_idx, loss.data[0])
                    print '==>>> batch index: {}/{}'.format(batch_idx+1, len(train_loader))
            print '==>>> batch index: {}/{}'.format(batch_idx+1, len(train_loader))

            if epoch > 1:
                print("-- Finish epoch {0:d}".format(i+1))

    @staticmethod
    def oneHot(target, num_classes):
        # oneHot encoding
        label = []
        for l in target:
                label.append([1 if i==l else 0 for i in range(num_classes)])
        return torch.FloatTensor(label)
