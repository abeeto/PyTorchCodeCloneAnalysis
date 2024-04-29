import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
from base_dataloader import BaseDataset, get_params, get_transform, normalize


class StereoDataloader_test(Dataset):
    __stereo = []

    def __init__(self, opt):
        self.opt = opt

        dirListing = os.listdir("/floyd/input/holopix_10k/stereo/")
        a = int(len(dirListing) * 0.8)
        for item in dirListing[a:]:
            if ".png" in item:
                self.__stereo.append("/floyd/input/holopix_10k/stereo/" + item)
            # if self.opt.ext_test:
            # 	os.system('find {0} -type f -name "*.jpg" > {0}/filename_test.txt'.format(self.opt.ext_test_in))
            # 	filename = self.opt.ext_filename_test
            # 	n_line = open(filename).read().count('\n')
            # 	self.__left = np.genfromtxt(filename, dtype=None)

        #
        # else:
        #
        # 	filename = self.opt.filename_test
        # 	dataroot = self.opt.dataroot
        # 	arrlenth = 66 + len(dataroot)
        # 	arrlen = '|S'+str(arrlenth)
        # 	arr = np.genfromtxt(filename, dtype=arrlen, delimiter=' ')
        # 	#transform_list = [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
        # 	#transform_list = [transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
        # 	#self.transforms = transforms.Compose(transform_list)
        # 	n_line = open(filename).read().count('\n')
        # 	#print(n_line)
        # 	for line in range(n_line):
        # 		self.__left.append(dataroot + arr[line][0])
        # 		# self.__right.append(dataroot + arr[line][1])

    def __getitem__(self, index):
        stereo_img = Image.open(self.__stereo[index]).convert('RGB')
        s = stereo_img.size
        img1 = stereo_img.crop((0, 0, s[0] / 2, s[1]))
        params = get_params(self.opt, img1.size)

        # img2 = Image.open(self.__right[index]).convert('RGB')

        transform = get_transform(self.opt, params)
        img1 = transform(img1)
        # img2 = transform(img2)

        # print('size(img1),size(img1)')
        # print(np.array(img1).shape,np.array(img1).shape)

        # img1 = torch.from_numpy(np.asarray(img1))
        # img2 = torch.from_numpy(np.asarray(img2))

        # print('type(img1),type(img1)')
        # print(type(img1),type(img1))

        # img1 = self.transforms(img1)
        # img2 = self.transforms(img2)

        input_dict = {'test_img': img1.cuda()}

        return input_dict

    def __len__(self):
        return len(self.__stereo)
