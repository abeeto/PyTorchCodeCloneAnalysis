import torch
import glob
import os
import json
import numpy as np
import cv2
#DEFAULT_SIZE = (256, 512)
DEFAULT_SIZE = (64, 128)


class HomographyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, train=True, size=DEFAULT_SIZE):
        # TODO
        self.flags = {'train': train, 'size': size}

        self.dataset_path = dataset_path
        self.json_lists = glob.glob(os.path.join(dataset_path, '*.json'))
        self.labels = []
        for json_list in self.json_lists:
            for line in open(json_list, 'r'):
                self.labels.append(json.loads(line))

        self.lanes = [lane['lanes'] for lane in self.labels]
        self.y_samples = [y_sample['h_samples'] for y_sample in self.labels]
        self.raw_files = [raw_file['raw_file'] for raw_file in self.labels]

        self.len = len(self.labels)

        self.img = np.zeros(size, np.uint8)
        self.segment_img = np.zeros(size, np.uint8)
        self.ins_img = self.segment_img.copy()
        print('LaneDataset init:', len(self.labels))

    def __getitem__(self, idx):
        # TODO
        lane_pts = [[(x, y) for (x, y) in zip(lane, self.y_samples[idx]) if x >= 0] for lane in self.lanes[idx]]

        self.img = cv2.imread(os.path.join(self.dataset_path, self.raw_files[idx]))
        self.height, self.width, _ = self.img.shape
        self.img = cv2.resize(self.img, (self.flags['size'][1],self.flags['size'][0]), interpolation=cv2.INTER_CUBIC)
        self.ins_img = np.zeros(self.flags['size'], dtype=np.uint8)


        ground_truth_trajectory=[]
        for i, lane_pt in enumerate(lane_pts):
            lane_pt_np = np.int32(lane_pt)
            lane_pt_np[:,0] //= (self.height//self.flags['size'][0])
            lane_pt_np[:,1] //= (self.width // self.flags['size'][1])
            cv2.polylines(self.ins_img, np.int32([lane_pt_np]), isClosed=False, color=i * 50 + 20, thickness=2)
            ground_truth_trajectory.append(np.int32([lane_pt_np]))


        return self.img, self.ins_img, ground_truth_trajectory

    def __len__(self):
        # TODO
        return self.len

if __name__=='__main__':
    import matplotlib.pyplot as plt
    hpd = HomographyPredictionDataset('/data/tusimple',train=True)
    img, inst_img, inst_array=hpd[0]
    f, axarr=plt.subplots(2,1)
    axarr[0].imshow(img)
    img_gray = np.mean(img, axis=2)
    axarr[1].imshow(img_gray + inst_img)
    plt.show()




