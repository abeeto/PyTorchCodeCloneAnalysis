# width, height
DEFAULT_SIZE = (256, 512)
import torch
from PIL import Image
import json
import os
import cv2
import numpy as np
import glob

class LaneDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, train=True, size=DEFAULT_SIZE):
        super().__init__()
        self.flags = {'train': train, 'size': size}

        self.dataset_path = dataset_path
        self.json_lists = glob.glob(os.path.join(dataset_path, '*.json'))
        self.labels = []
        for json_list in self.json_lists:
          for line in open(json_list,'r'):
            self.labels.append(json.loads(line))

        self.lanes = [lane['lanes'] for lane in self.labels]
        self.y_samples = [y_sample['h_samples'] for y_sample in self.labels]
        self.raw_files = [raw_file['raw_file'] for raw_file in self.labels]

        self.len = len(self.labels)

        self.img = np.zeros((size[1], size[0]), np.uint8)
        self.segment_img = np.zeros((size[1], size[0]), np.uint8)
        self.ins_img = self.segment_img.copy()
        print('LaneDataset init:', len(self.labels))

    def get_lane_image(self, idx):
        lane_pts = [[(x, y) for (x, y) in zip(lane, self.y_samples[idx]) if x >= 0] for lane in self.lanes[idx]]
        #while len(lane_pts) < self.n_seg:
        #    lane_pts.append(list())

        self.img = cv2.imread(os.path.join(self.dataset_path, self.raw_files[idx]))
        self.height, self.width, _ = self.img.shape
        self.segment_img = np.zeros((self.height, self.width), dtype=np.uint8)
        self.ins_img = np.zeros((self.height, self.width), dtype=np.uint8)

        for i, lane_pt in enumerate(lane_pts):
            cv2.polylines(self.segment_img, np.int32([lane_pt]), isClosed=False, color=(1), thickness=10)
            #gt = np.zeros((self.height, self.width), dtype=np.uint8)
            #gt = cv2.polylines(gt, np.int32([lane_pt]), isClosed=False, color=(1), thickness=5)

            cv2.polylines(self.ins_img, np.int32([lane_pt]), isClosed=False, color=i * 50 + 20, thickness=10)
            #self.ins_img = np.concatenate([self.ins_img, gt[np.newaxis]])
        #cv2.imwrite('ins_img_dbg.bmp', self.ins_img)
        pass

    def image_resize(self):
        ins = []
        # resize之前：self.img为720*1280*3、self.label_image为720*1280
        self.img = cv2.resize(self.img, (self.flags['size'][1],self.flags['size'][0]), interpolation=cv2.INTER_CUBIC)
        self.segment_img = cv2.resize(self.segment_img, (self.flags['size'][1],self.flags['size'][0]), interpolation=cv2.INTER_NEAREST)

        self.ins_img = cv2.resize(self.ins_img, (self.flags['size'][1], self.flags['size'][0]),
                                      interpolation=cv2.INTER_NEAREST)
        #for i in range(len(self.ins_img)):
        #    dst = cv2.resize(self.ins_img[i], (self.flags['size'][1],self.flags['size'][0]), interpolation=cv2.INTER_CUBIC)
        #    ins.append(dst)


        #self.ins_img = np.array(ins, dtype=np.uint8)

    def preprocess(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2LAB)
        img_plane = cv2.split(img)
        img_plane=list(img_plane)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_plane[0] = clahe.apply(img_plane[0])
        img = cv2.merge(img_plane)
        self.img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    def __getitem__(self, idx):
        # TODO
        self.get_lane_image(idx)
        self.image_resize()
        self.preprocess()

        self.img = np.array(np.transpose(self.img, (2, 0, 1)), dtype=np.float32)
        if self.flags['train']:
            self.segment_img = np.array(self.segment_img, dtype=np.float32)
            self.ins_img = np.array(self.ins_img, dtype=np.float32)
            return torch.Tensor(self.img), torch.LongTensor(self.segment_img), torch.Tensor(self.ins_img)
        else:

            return torch.Tensor(self.img)
        #return image, segmentation_image, instance_image  # 1 x H x W [[0, 1], [2, 0]]

    def __len__(self):
        # TODO
        return self.len


if __name__=='__main__':
    ldata = LaneDataset('/data/tusimple')
    ldata.get_lane_image(0)
    ldata.image_resize()
    ldata.preprocess()
    cv2.imshow('org_img', ldata.img)
    cv2.imshow('segment_img', ldata.segment_img*255)
    cv2.waitKey(0)
    cv2.imshow('inst_img', ldata.ins_img)
