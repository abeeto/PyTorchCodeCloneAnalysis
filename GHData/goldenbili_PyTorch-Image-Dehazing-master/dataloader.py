"""
dataloader Ver1
"""

import glob
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

random.seed(1143)


def populate_train_list(orig_images_path):
    tmp_img_name = []
    image_list_orig = glob.glob(orig_images_path + "*.png")
    for image in image_list_orig:
        image = image.split("/")[-1]
	# print('image path:' + image)
        tmp_img_name.append(image)

    train_keys = []
    val_keys = []
    len_img = len(tmp_img_name)
    for i in range(len_img):
        if i < len_img * 9 / 10:
            train_keys.append(tmp_img_name[i])
        else:
            val_keys.append(tmp_img_name[i])

    train_list = []
    val_list = []
    for train_image in train_keys:
        train_list.append(orig_images_path + train_image)
    for valid_image in val_keys:
        val_list.append(orig_images_path + valid_image)

    # 亂序
    random.shuffle(train_list)
    random.shuffle(val_list)

    return train_list, val_list


def getbgr(img):
    mode = img.mode
    if mode == 'RGBA':
        r, g, b, a = img.split()

    elif mode == 'RGB':
        r, g, b = img.split()
    else:
        b = [], g = [], r = []

    return r, g, b


def bgr2yuv(data_orig, bk_w, bk_h):
    R, G, B = getbgr(data_orig)
    width = data_orig.width
    height = data_orig.height
    '''
	print('In the bgr2yuv')
	print('width:')
	print(width)
	print('height')
	print(height)
	'''

    R = np.reshape(list(R.getdata()), (height, width))
    G = np.reshape(list(G.getdata()), (height, width))
    B = np.reshape(list(B.getdata()), (height, width))

    list_yuv444 = []
    list_yuv420 = []
    list_rgb = []
    # original
    index_bk = 0
    for i in range(0, height, bk_h):
        for j in range(0, width, bk_w):

            # blocks
            # ------------------------------------------#
            # 二維
            im_new_Y_Blk = np.full((bk_h, bk_w), np.inf)
            im_new_U_Blk = np.full((bk_h, bk_w), np.inf)
            im_new_V_Blk = np.full((bk_h, bk_w), np.inf)

            im_new_R_Blk = np.full((bk_h, bk_w), np.inf)
            im_new_G_Blk = np.full((bk_h, bk_w), np.inf)
            im_new_B_Blk = np.full((bk_h, bk_w), np.inf)

            y_blk = 0
            x_blk = 0
            for y in range(i, i + bk_h):
                for x in range(j, j + bk_w):

                    # if not enough one block ... (跳掉)
                    # --------------------------#
                    if i + bk_h > height or j + bk_w > width:
                        continue
                    # --------------------------#

                    # Get value
                    # --------------------------#
                    index_all = y * bk_w + x
                    r = R[y][x]
                    g = G[y][x]
                    b = B[y][x]
                    # --------------------------#

                    # Block Setting
                    # --------------------------#
                    im_new_Y_Blk[y_blk, x_blk] = int(0.299 * r + 0.587 * g + 0.114 * b)
                    im_new_U_Blk[y_blk, x_blk] = int(-0.1687 * r - 0.3313 * g + 0.5 * b + 128)
                    im_new_V_Blk[y_blk, x_blk] = int(0.5 * r - 0.4187 * g + - 0.0813 * b + 128)
                    # --------------------------#
                    # Block rgb
                    # --------------------------#
                    im_new_R_Blk[y_blk, x_blk] = r
                    im_new_G_Blk[y_blk, x_blk] = g
                    im_new_B_Blk[y_blk, x_blk] = b
                    # --------------------------#

                    x_blk = x_blk + 1

                y_blk = y_blk + 1
                x_blk = 0
            # ------------------------------------------#

            # 420平均
            step_x = 2
            step_y = 2

            im_new_U_Blk_Ori = im_new_U_Blk.copy()
            im_new_V_Blk_Ori = im_new_V_Blk.copy()
            for y in range(0, bk_h, step_y):
                for x in range(0, bk_w, step_x):
                    # 存成一組
                    mean_U = np.mean(im_new_U_Blk[y:y + step_y, x:x + step_x])
                    mean_V = np.mean(im_new_V_Blk[y:y + step_y, x:x + step_x])

                    im_new_U_Blk[y:y + step_y, x:x + step_x].fill(mean_U)
                    im_new_V_Blk[y:y + step_y, x:x + step_x].fill(mean_V)

            # 三維
            # 422
            # Array_blk_420 = [im_new_Y_Blk, im_new_U_Blk, im_new_V_Blk]
            # yuv_420.append(Array_blk_420)
            # yuv_420 = np.append(im_new_Y_Blk,im_new_U_Blk,im_new_V_Blk,axis=0)
            yuv_444 = np.zeros(shape=(3, bk_h, bk_w))
            yuv_420 = np.zeros(shape=(3, bk_h, bk_w))
            rgb_ori = np.zeros(shape=(3, bk_h, bk_w))

            yuv_420[0] = im_new_Y_Blk
            yuv_420[1] = im_new_U_Blk
            yuv_420[2] = im_new_V_Blk
            list_yuv420.append(yuv_420)

            rgb_ori[0] = im_new_R_Blk
            rgb_ori[1] = im_new_G_Blk
            rgb_ori[2] = im_new_B_Blk
            list_rgb.append(rgb_ori)

            # 444
            # Array_blk_444 = [im_new_Y_Blk,im_new_U_Blk_Ori, im_new_V_Blk_Ori]
            # yuv_444.append(Array_blk_444)
            # yuv_444 = np.append(im_new_Y_Blk,im_new_U_Blk_Ori, im_new_V_Blk_Ori, axis=0)
            # yuv_444 = np.append(im_new_Y_Blk,im_new_U_Blk_Ori,axis=0)
            # yuv_444 = np.append(yuv_444,im_new_V_Blk_Ori,axis=0)
            yuv_444[0] = im_new_Y_Blk
            yuv_444[1] = im_new_U_Blk_Ori
            yuv_444[2] = im_new_V_Blk_Ori
            list_yuv444.append(yuv_444)

    return list_yuv444, list_yuv420, list_rgb


class dehazing_loader(data.Dataset):

    def __init__(self, orig_images_path, mode, resize, bk_width, bk_height, btest):
        self.train_list, self.val_list = populate_train_list(orig_images_path)
        self.resize = resize
        self.bkW = bk_width
        self.bkH = bk_height
        self.mode = mode
        self.bTest = btest

        if mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))
        self.temp_count = 0

    def __getitem__(self, index):
        data_orig_path = self.data_list[index]
        bkW = self.bkW
        bkH = self.bkH
        # load image
        data_orig = Image.open(data_orig_path)
        bl_num_width = data_orig.width / bkW
        bl_num_height = data_orig.height / bkH

        if self.resize == 1:
            data_orig = data_orig.resize((640, 480), Image.ANTIALIAS)
            bl_num_width = 640 / bkW
            bl_num_height = 480 / bkH

        if data_orig.width == 640 and data_orig.height == 480:
            data_orig.save("fileout" + str(self.temp_count) + ".png")
            self.temp_count = self.temp_count + 1
        '''
		data_orig = (np.asarray(data_orig)/255.0)

		data_orig = torch.from_numpy(data_orig).float()

		data_orig = data_orig.permute(2, 0, 1)

		return data_orig
		'''

        # data_orig to yuv444, yuv420
        data_yuv444, data_yuv420, data_rgb = bgr2yuv(data_orig, bkW, bkH)
        '''
		print('size with data_yuv444:')
		print(len(data_yuv444))
		print('size with data_yuv420:')
		print(len(data_yuv420))
		print('size with data_rgb:')
		print(len(data_rgb))
		'''

        list_tensor_yuv444 = []
        list_tensor_yuv420 = []
        list_tensor_rgb = []
        for yuv444 in data_yuv444:
            # 測試板 外面量化
            '''
			if self.mode == 'train':
				yuv444 = (np.asarray(yuv444)/255.0)
			else:
				print('Is bTest and valid ')
			'''
            yuv444 = (np.asarray(yuv444) / 255.0)
            yuv444_unit = torch.from_numpy(yuv444).float()
            list_tensor_yuv444.append(yuv444_unit)

        for yuv420 in data_yuv420:
            yuv420 = (np.asarray(yuv420) / 255.0)
            yuv420_unit = torch.from_numpy(yuv420).float()
            list_tensor_yuv420.append(yuv420_unit)

        for rgb in data_rgb:
            rgb = (np.asarray(rgb) / 255.0)
            rgb_unit = torch.from_numpy(rgb).float()
            list_tensor_rgb.append(rgb_unit)

        # list_tensor_yuv420.append(torch.from_numpy(yuv420).float())

        return list_tensor_yuv444, list_tensor_yuv420, list_tensor_rgb, bl_num_width, bl_num_height, data_orig_path

    def __len__(self):
        return len(self.data_list)

    '''
	data_orig = (np.asarray(data_orig)/255.0)
	data_hazy = (np.asarray(data_hazy)/255.0)
	return data_orig.permute(2,0,1), data_hazy.permute(2,0,1)
	'''

    # ori-process
    '''
		data_hazy = Image.open(data_hazy_path)
		#

		# ori-process
		# resize to 640*480 + 平滑filter
		data_orig = data_orig.resize((480,640), Image.ANTIALIAS)
		data_hazy = data_hazy.resize((480,640), Image.ANTIALIAS)
		#
		data_orig = np.asarray(data_orig)
		data_hazy = np.asarray(data_hazy)
		#
		data_orig = (data_orig / 255.0)
		data_hazy = (data_hazy / 255.0)

		data_orig = torch.from_numpy(data_orig).float()
		data_hazy = torch.from_numpy(data_hazy).float()

		data_orig = data_orig.permute(2, 0, 1)
		data_hazy = data_hazy.permute(2, 0, 1)

		return data_orig, data_hazy


	'''
