import json
import os

import cv2
import numpy as np
import torch
from imageio import imread
from PIL import Image
from torchvision import transforms

import lib.utils.data as torchdata
from broden_dataset_utils.joint_dataset import broden_dataset


# Round x to the nearest multiple of p and x' >= x
def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p


def decodeRG(im):
    return (im[:, :, 0] // 10) * 256 + im[:, :, 1]


def encodeRG(channel):
    result = np.zeros(channel.shape + (3,), dtype=np.uint8)
    result[:, :, 0] = (channel // 256) * 10
    result[:, :, 1] = channel % 256
    return result


def uint16_imresize(seg, shape):
    return decodeRG(np.array(Image.fromarray(encodeRG(seg)).resize(shape, PIL.Image.NEAREST)))


class TrainDataset(torchdata.Dataset):
    def __init__(self, records, source_idx, opt, max_sample=-1, batch_per_gpu=1):
        self.imgSize = opt.imgSize
        self.imgMaxSize = opt.imgMaxSize
        self.random_flip = opt.random_flip

        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])
        ])

        self.list_sample = records
        self.source_idx = source_idx

        self.if_shuffled = False
        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample)  # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample)  # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSize, list):
            this_short_size = np.random.choice(self.imgSize)
        else:
            this_short_size = self.imgSize

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_resized_size = np.zeros((self.batch_per_gpu, 2), np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(this_short_size / min(img_height, img_width), self.imgMaxSize / max(img_height, img_width))
            img_resized_height, img_resized_width = img_height * this_scale, img_width * this_scale
            batch_resized_size[i, :] = img_resized_height, img_resized_width
        batch_resized_height = np.max(batch_resized_size[:, 0])
        batch_resized_width = np.max(batch_resized_size[:, 1])

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_resized_height = int(round2nearest_multiple(batch_resized_height, self.padding_constant))
        batch_resized_width = int(round2nearest_multiple(batch_resized_width, self.padding_constant))

        assert self.padding_constant >= self.segm_downsampling_rate, \
            'padding constant must be equal or large than segm downsamping rate'

        batch_images = torch.zeros((self.batch_per_gpu, 3, batch_resized_height, batch_resized_width))
        batch_objs = torch.zeros((self.batch_per_gpu, batch_resized_height // self.segm_downsampling_rate,
                                  batch_resized_width // self.segm_downsampling_rate)).long()
        batch_valid_obj = torch.zeros(self.batch_per_gpu).long()
        batch_parts = torch.zeros((self.batch_per_gpu, broden_dataset.nr_object_with_part,
                                   batch_resized_height // self.segm_downsampling_rate,
                                   batch_resized_width // self.segm_downsampling_rate)).long()
        batch_valid_parts = torch.zeros((self.batch_per_gpu, broden_dataset.nr_object_with_part)).long()
        batch_scene_labels = torch.zeros(self.batch_per_gpu).long()
        batch_material = torch.zeros((self.batch_per_gpu, batch_resized_height // self.segm_downsampling_rate,
                                      batch_resized_width // self.segm_downsampling_rate)).long()
        batch_valid_mat = torch.zeros(self.batch_per_gpu).long()

        for i in range(self.batch_per_gpu):

            data = broden_dataset.resolve_record(batch_records[i])

            img = data['img']
            seg_obj = data["seg_obj"]
            valid_obj = data["valid_obj"]
            seg_part = data["batch_seg_part"]
            valid_part = data["valid_part"]
            scene_label = data["scene_label"]
            seg_material = data["seg_material"]
            valid_mat = data["valid_mat"]

            # scene
            batch_scene_labels[i] = int(scene_label)

            # random flip img obj part material
            if self.random_flip:
                random_flip = np.random.choice([0, 1])
                if random_flip == 1:
                    img = cv2.flip(img, 1)
                    seg_obj = cv2.flip(seg_obj, 1)
                    seg_part = np.flip(seg_part, 2)
                    seg_material = cv2.flip(seg_material, 1)

            # img
            img = np.array(Image.fromarray(img).resize((batch_resized_size[i, 0], batch_resized_size[i, 1]), PIL.Image.BILINEAR))
            #img = imresize(img, (batch_resized_size[i, 0], batch_resized_size[i, 1]), interp='bilinear')
            img = img.astype(np.float32)[:, :, ::-1]  # RGB to BGR!!!
            img = img.transpose((2, 0, 1))
            img = self.img_transform(torch.from_numpy(img.copy()))
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img

            # object and part
            if valid_obj:
                batch_valid_obj[i] = valid_obj

                # object
                segm = uint16_imresize(seg_obj, (batch_resized_size[i, 0], batch_resized_size[i, 1]))
                segm_rounded_height = round2nearest_multiple(segm.shape[0], self.padding_constant)
                segm_rounded_width = round2nearest_multiple(segm.shape[1], self.padding_constant)
                segm_rounded = np.zeros((segm_rounded_height, segm_rounded_width), dtype='uint16')
                segm_rounded[:segm.shape[0], :segm.shape[1]] = segm
                segm = uint16_imresize(segm_rounded,
                                       (segm_rounded.shape[0] // self.segm_downsampling_rate,
                                        segm_rounded.shape[1] // self.segm_downsampling_rate))
                batch_objs[i][:segm.shape[0], :segm.shape[1]] = torch.from_numpy(np.array(segm, dtype=np.int32))

                # part
                if np.sum(valid_part) == 0:
                    continue

                parts_resized = []
                for j in range(broden_dataset.nr_object_with_part):
                    parts_resized.append(img = np.array(Image.fromarray(seg_part[j]).resize((batch_resized_size[i, 0], batch_resized_size[i, 1]), PIL.Image.NEAREST)))
                    #parts_resized.append(imresize(seg_part[j], (batch_resized_size[i, 0], batch_resized_size[i, 1]), interp='nearest'))
                for j in range(broden_dataset.nr_object_with_part):
                    if not valid_part[j]:
                        continue
                    part_rounded = np.zeros((segm_rounded_height, segm_rounded_width), dtype='uint8')
                    part_rounded[:parts_resized[j].shape[0], :parts_resized[j].shape[1]] = parts_resized[j]
                    part = np.array(Image.fromarray(part_rounded).resize((part_rounded.shape[0], part_rounded.shape[1]), PIL.Image.NEAREST))
                    #part = imresize(part_rounded,
                     #               (part_rounded.shape[0] // self.segm_downsampling_rate,
                     #                part_rounded.shape[1] // self.segm_downsampling_rate), interp='nearest')
                    batch_parts[i][j][:part.shape[0], :part.shape[1]] = torch.from_numpy(part.copy())
                    # NOTE: part seg might disappear after resize.
                    if len(np.unique(part)) > 1:
                        batch_valid_parts[i][j] = 1
            # material
            if valid_mat:
                batch_valid_mat[i] = valid_mat
                segm = np.array(Image.fromarray(seg_material).resize((batch_resized_size[i, 0], batch_resized_size[i, 1]), PIL.Image.NEAREST))
                #segm = imresize(seg_material,
                 #               (batch_resized_size[i, 0], batch_resized_size[i, 1]), interp='nearest')
                segm_rounded_height = round2nearest_multiple(segm.shape[0], self.padding_constant)
                segm_rounded_width = round2nearest_multiple(segm.shape[1], self.padding_constant)
                segm_rounded = np.zeros((segm_rounded_height, segm_rounded_width), dtype='uint8')
                segm_rounded[:segm.shape[0], :segm.shape[1]] = segm
                segm = np.array(Image.fromarray(segm_rounded).resize((segm_rounded.shape[0], segm_rounded.shape[1]), PIL.Image.NEAREST))
                #segm = imresize(segm_rounded,
                 #               (segm_rounded.shape[0] // self.segm_downsampling_rate,
                  #               segm_rounded.shape[1] // self.segm_downsampling_rate), interp='nearest')
                batch_material[i][:segm.shape[0], :segm.shape[1]] = torch.from_numpy(segm.copy())

        # use compressed part segm
        # TODO(LYC):: remove compression
        batch_parts = torch.sum(batch_parts, dim=1)

        # convert numpy array to torch tensor

        output = dict(
            img=batch_images,
            seg_object=batch_objs,
            valid_object=batch_valid_obj,
            seg_part=batch_parts,
            valid_part=batch_valid_parts,
            scene_label=batch_scene_labels,
            seg_material=batch_material,
            valid_material=batch_valid_mat,
            source_idx=torch.tensor(self.source_idx),
        )

        return output

    def __len__(self):
        return int(1e6)  # It's a fake length due to the trick that every loader maintains its own list
        # return self.num_sampleclass


class ValDataset(torchdata.Dataset):
    def __init__(self, records, opt, max_sample=-1, start_idx=-1, end_idx=-1):
        self.imgSize = opt.imgSize
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])
        ])

        self.list_sample = records

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]

        if start_idx >= 0 and end_idx >= 0:  # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def __getitem__(self, index):
        data = broden_dataset.resolve_record(self.list_sample[index])
        output = {}

        # image
        img = data['img']
        img = img[:, :, ::-1]  # BGR to RGB!!!
        ori_height, ori_width, _ = img.shape
        img_resized_list = []
        for this_short_size in self.imgSize:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_height = round2nearest_multiple(target_height, self.padding_constant)
            target_width = round2nearest_multiple(target_width, self.padding_constant)

            # resize
            img_resized = cv2.resize(img.copy(), (target_width, target_height))

            # image to float
            img_resized = img_resized.astype(np.float32)
            img_resized = img_resized.transpose((2, 0, 1))
            img_resized = self.img_transform(torch.from_numpy(img_resized))

            img_resized_list.append(img_resized)
        output['img_resized_list'] = [x.contiguous() for x in img_resized_list]
        output['original_img'] = img

        # object
        output['seg_object'] = torch.from_numpy(
            data["seg_obj"].astype(np.int32)).long().contiguous()
        output['valid_object'] = torch.tensor(int(data['valid_obj'])).long()

        # part
        output['seg_part'] = torch.from_numpy(
            np.sum(data["batch_seg_part"], axis=0).astype(np.uint8)).long().contiguous()
        output['valid_part'] = torch.from_numpy(data['valid_part'].astype(np.uint8)).long()

        # scene
        output['scene_label'] = torch.tensor(int(data['scene_label']))

        # material
        output['seg_material'] = torch.from_numpy(data['seg_material']).contiguous()
        output['valid_material'] = torch.tensor(int(data['valid_mat'])).long()

        return output

    def __len__(self):
        return self.num_sample


class TestDataset(torchdata.Dataset):
    def __init__(self, odgt, opt, max_sample=-1):
        self.imgSize = opt.imgSize
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])
        ])

        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = this_record['fpath_img']
        img = imread(image_path, as_gray=False, pilmode='RGB')
        img = img[:, :, ::-1]  # BGR to RGB!!!

        ori_height, ori_width, _ = img.shape

        img_resized_list = []
        for this_short_size in self.imgSize:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_height = round2nearest_multiple(target_height, self.padding_constant)
            target_width = round2nearest_multiple(target_width, self.padding_constant)

            # resize
            img_resized = cv2.resize(img.copy(), (target_width, target_height))

            # image to float
            img_resized = img_resized.astype(np.float32)
            img_resized = img_resized.transpose((2, 0, 1))
            img_resized = self.img_transform(torch.from_numpy(img_resized))

            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        # segm = torch.from_numpy(segm.astype(np.int)).long()

        # batch_segms = torch.unsqueeze(segm, 0)

        # batch_segms = batch_segms - 1 # label from -1 to 149
        output = dict()
        output['img_ori'] = img.copy()
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        # output['seg_label'] = batch_segms.contiguous()
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample
