import torch
import cv2
import numpy as np
import glob as glob
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms):
        self.transforms = transforms
        self.dir_path = dir_path 
        self.height = height
        self.width = width
        self.classes = classes
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        self.file_names = [(image_path.split("/")[-1]).split(".")[0] for image_path in self.image_paths]

    def __getitem__(self, idx):
        boxes = []
        labels = []     
        image = cv2.imread(self.dir_path+self.file_names[idx]+".jpg")
        image_height,image_width,_ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        with open(self.dir_path+self.file_names[idx]+".txt") as f:
            for line in f:
                bbox_coordinates = line.split(" ")
                labels.append(int(bbox_coordinates[0]))
                center_x = float(bbox_coordinates[1]) * image_width
                center_y = float(bbox_coordinates[2]) * image_height
                width_yolo = float(bbox_coordinates[3]) *image_width
                height_yolo = float(bbox_coordinates[4]) *image_height
                x_min = int(center_x - (width_yolo / 2))
                x_max = int(center_x + (width_yolo / 2))
                y_min = int(center_y - (height_yolo / 2))
                y_max = int(center_y + (height_yolo / 2))
                xmin_final = (x_min/image_width) * self.width
                xmax_final = (x_max/image_width) * self.width
                ymin_final = (y_min/image_height) * self.height
                yamx_final = (y_max/image_height) * self.height
                boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
                
        return image_resized, target

    def __len__(self):
        return len(self.file_names)