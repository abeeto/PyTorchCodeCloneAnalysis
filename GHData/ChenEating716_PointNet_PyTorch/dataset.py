# Author: Yiting CHEN
# Email: chenyiting@whu.edu.cn
# Time: 2022/02/26


import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import torch
from tqdm import tqdm
import numpy as np
import open3d as o3d


class ModelNet40(Dataset):
    def __init__(self, root_dir, train):
        super(ModelNet40, self).__init__()

        self._classes = []
        self._pcds = []
        self._labels = []

        # 0: train / 1: test
        self._train = train
        self.load(root_dir)

    def read_pcd_from_txt(self, file):
        """load point cloud from .txt file

        Args:
            file (file_path): path

        Returns:
            _type_: point cloud in np.array
        """
        with open(file, 'r') as f:
            pcd = []
            for point in f:
                pt = list(map(float, point.split(',')))[:3]
                pcd.append(pt)

        pcd = np.asarray(pcd)
        return pcd

    def read_file_names_from_file(self, file):
        files = []
        with open(file, 'r') as f:
            for line in f:
                files.append(line.split('\n')[0])

        return files

    def pcd_visualization(self, pcd):
        """visualize point cloud

        Args:
            pcd (np.array): point cloud in np.array
        """
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcd)
        o3d.visualization.draw_geometries([point_cloud])

    def pcd_rotation(self, pcd, z_angle):
        """rotate numpy point cloud 

        Args:
            pcd (np.array): point cloud
            z_angle ([0, 2*Pi]): angle

        Returns:
            np.array: point cloud
        """
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcd)
        xyz = np.array([0, 0, z_angle])
        R = o3d.geometry.get_rotation_matrix_from_xyz(xyz)
        point_cloud = point_cloud.rotate(R)
        pcd = np.asarray(point_cloud.points)
        return pcd

    def pcd_normalization(self, pcd):
        center_point = np.mean(pcd, axis=0)
        normalized_pcd = pcd - center_point

        return normalized_pcd

    def classes(self):
        return self._classes

    def __len__(self):
        return len(self._pcds)

    def __getitem__(self, index):
        pcd, label = self._pcds[index], self._labels[index]

        pcd = self.read_pcd_from_txt(pcd)

        # normalize
        pcd = self.pcd_normalization(pcd)

        # rotation
        z_angle = np.random.rand(1)[0] * 2 * math.pi
        pcd = self.pcd_rotation(pcd, z_angle)

        # jitter
        pcd += np.random.normal(0, 0.02, size=pcd.shape)
        pcd = torch.Tensor(pcd.T)

        # label
        l_label = [0 for _ in range(len(self._classes))]
        l_label[self._classes.index(label)] = 1
        label = torch.Tensor(l_label)

        return pcd, label

    def load(self, root_dir):
        files_list = os.listdir(root_dir)
        files = []
        for f in files_list:
            if self._train == 0:
                if f == 'modelnet40_train.txt':
                    files = self.read_file_names_from_file(root_dir + '/' + f)
            elif self._train == 1:
                if f == 'modelnet40_test.txt':
                    files = self.read_file_names_from_file(root_dir + '/' + f)
            if f == 'modelnet40_shape_names.txt':
                self._classes = self.read_file_names_from_file(root_dir + '/' + f)
        tmp_classes = []
        progress = tqdm(total=len(files))

        for file in files:
            num = file.split("_")[-1]
            pcd_label = file.split("_" + num)[0]

            if pcd_label not in tmp_classes:
                tmp_classes.append(pcd_label)
            pcd_file_path = root_dir + '/' + pcd_label + '/' + file + '.txt'
            # pcd_file = self.read_pcd_from_txt(pcd_file_path)

            self._pcds.append(pcd_file_path)
            self._labels.append(pcd_label)

            progress.update(1)

        if self._train == 0:
            print("loading {} trianing point cloud files".format(len(self._labels)))
        elif self._train == 1:
            print("loading {} testing point cloud files".format(len(self._labels)))


if __name__ == '__main__':
    MN_data = ModelNet40('./dataset/modelnet40_normal_resampled', 1)
    train_loader = DataLoader(MN_data, batch_size=32, shuffle=True)

    cnt = 0
    for pcd, labels in train_loader:
        print(labels.argmax(axis=1).shape)
        cnt += 1
        if cnt == 3:
            break

    # pcd = MN_data.read_pcd_from_txt('./dataset/modelnet40_normal_resampled/airplane/airplane_0001.txt')
    # pcd = MN_data.pcd_normalization(pcd)
    # pcd = MN_data.pcd_rotation(pcd, math.pi/2)
    # MN_data.pcd_visualization(pcd)

