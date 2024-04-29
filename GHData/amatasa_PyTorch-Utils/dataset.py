import os
import cv2
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import torch.nn.functional as F

from torch.utils.data import Dataset

class CASIA(Dataset):
    def __init__(self, seq_dir, label, seq_type, view, cuboid_len, phase="Train"):
        self.seq_dir = seq_dir
        self.label = label
        self.seq_type = seq_type
        self.view = view
        self.cuboid_len = cuboid_len

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        video = list()

        for filename in os.listdir(self.seq_dir[index]):
            if filename.split(".")[0][-1] != 'c':
                path = os.path.join(self.seq_dir[index], filename)
                frame = torch.Tensor(cv2.imread(path, 0))
                if frame.shape[0] == 240 and frame.shape[1] == 320:
                    video.append(frame)

        if len(video) < self.cuboid_len:
            return None, None, None, None

        video = torch.stack(video)
        video = video.unsqueeze(1)

        cuboids = video.split(self.cuboid_len)

        if len(cuboids) >= 1:
            if len(cuboids[-1]) != self.cuboid_len:
                cuboids = cuboids[:-1]
            print(len(cuboids))
            cuboids = self.crop(cuboids)
            
            if cuboids == None:
                return None, None, None, None
            
            labels = [int(self.label[index])]*len(cuboids)
            types = [self.seq_type[index]]*len(cuboids)
            views = [self.view[index]]*len(cuboids)

        return cuboids, labels, types, views

        

    def crop(self, videos):
        cuboids = list()

        height = videos[-1].shape[2]
        width = videos[-1].shape[3]

        for i, cuboid in enumerate(videos):
            boxes = list()
            first = cuboid[0]
            last = cuboid[-1]

            if first.sum(dim=2).sum(dim=1) == 0:
                for frame in cuboid:
                    if first.sum(dim=2).sum(dim=1) != 0:
                        print("Fixed blank starting")
                        first = frame
                        break

            if last.sum(dim=2).sum(dim=1) == 0:
                for frame in reversed(cuboid):
                    if last.sum(dim=2).sum(dim=1) != 0:
                        print("Fixed blank ending")
                        last = frame
                        break

            #print(first.sum(dim=2).sum(dim=1), last.sum(dim=2).sum(dim=1))

            for frame in [first, last]:
                frame = frame.squeeze(0).numpy()

                horizontal_indicies = np.where(np.any(frame, axis=0))[0]
                vertical_indicies = np.where(np.any(frame, axis=1))[0]

                if horizontal_indicies.shape[0]:
                    x1, x2 = horizontal_indicies[[0, -1]]
                    y1, y2 = vertical_indicies[[0, -1]]

                    if (x2-x1) % 2 != 0:
                        x2 = x2-1
                    if (y2-y1) % 2 != 0:
                        y2 = y2-1

                    boxes.append([x1, x2, y1, y2])

                #print("Cropped Frame Dimensions: [{}, {}, {}, {}]".format(*boxes[-1]))

            if len(boxes) < 2:
                return None

            overall = [
                min(boxes[0][0], boxes[1][0]), #x1
                max(boxes[0][1], boxes[1][1]), #x2
                min(boxes[0][2], boxes[1][2]), #y1
                max(boxes[0][3], boxes[1][3]), #y2
                ]

            #print("\nOverall Video Crop Dimensions: [{}, {}, {}, {}]".format(*overall))
            #print("Pre-Cropped Cuboid Shape: {}".format(cuboid.shape))

            cuboid = cuboid[:,:,overall[2]:overall[3],overall[0]:overall[1]] 

            # up/downsample to 56x56 then pad to 64x64
            size = 56

            scale_y = cuboid.shape[2] / size
            scale_x = cuboid.shape[3] / size

            max_scale = max(scale_y, scale_x)

            #print("x-scale: {} y-scale: {}".format(scale_x, scale_y))

            #print("scaled shape: [{},{}]".format(cuboid.shape[2]/max_scale, cuboid.shape[3]/max_scale))

            cuboid = F.interpolate(cuboid, size=(int(cuboid.shape[2]/max_scale), int(cuboid.shape[3]/max_scale)))

            #print("cuboid shape {}".format(cuboid.shape))

            # pad cuboid
            y_pad = int(64 - cuboid.shape[2])
            x_pad = int(64 - cuboid.shape[3])

            # y padding
            if y_pad % 2 == 0:
                y_pad = y_pad / 2
                y_pad = torch.zeros(self.cuboid_len, 1, int(y_pad), int(cuboid.shape[3]))
                cuboid = torch.cat([y_pad, cuboid, y_pad], axis=2)
                
            else:
                y_pad_1 = math.floor(y_pad / 2)
                y_pad_2 = y_pad_1 + 1

                y_pad_1 = torch.zeros(self.cuboid_len, 1, int(y_pad_1), int(cuboid.shape[3]))
                y_pad_2 = torch.zeros(self.cuboid_len, 1, int(y_pad_2), int(cuboid.shape[3]))

                cuboid = torch.cat([y_pad_1, cuboid, y_pad_2], axis=2)

            # x padding
            if x_pad % 2 == 0:
                x_pad = x_pad / 2
                x_pad = torch.zeros(self.cuboid_len, 1, int(cuboid.shape[2]), int(x_pad))
                cuboid = torch.cat([x_pad, cuboid, x_pad], axis=3)
            else:
                x_pad_1 = math.floor(x_pad / 2)
                x_pad_2 = x_pad_1 + 1

                x_pad_1 = torch.zeros(self.cuboid_len, 1, int(cuboid.shape[2]), int(x_pad_1))
                x_pad_2 = torch.zeros(self.cuboid_len, 1, int(cuboid.shape[2]), int(x_pad_2))

                cuboid = torch.cat([x_pad_1, cuboid, x_pad_2], axis=3)


            #print("cuboid shape {}".format(cuboid.shape))
            '''  
            img = [cuboid[j,0,:,:] for j in range(cuboid.shape[0])] # some array of images
            frames = [] # for storing the generated images

            fig, ax = plt.subplots()
            container = []

            for j in range(len(img)):
                image = ax.imshow(img[j], cmap="gray")
                title = ax.text(0.5,1.05,"Cuboid {}: Frame {:2d}".format(i+1, j+1), 
                                size=plt.rcParams["axes.titlesize"],
                                ha="center", transform=ax.transAxes, )
                container.append([image, title])

            ani = animation.ArtistAnimation(fig, container, interval=200, blit=False)

            plt.show()
            '''

            # random crop
            x_shift = torch.LongTensor(1).random_(0, 8)
            y_shift = torch.LongTensor(1).random_(0, 8)
            if self.cuboid_len == 16:
                t_shift = 0
            else:
                t_shift = torch.LongTensor(1).random_(0,self.cuboid_len-16)

            cuboid = cuboid[t_shift:t_shift+16,:,y_shift:y_shift+56,x_shift:x_shift+56]
            cuboids.append(cuboid)

        cuboids = torch.stack(cuboids).permute(0,2,1,3,4)

        return cuboids