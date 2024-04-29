import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class detection_dataset(Dataset):
    def __init__(self, data_file: str, cls_file: str, hw=(300, 300)):
        assert os.path.isfile(data_file), f"{data_file}: 文件不存在"
        assert os.path.isfile(cls_file), f"{cls_file}: 文件不存在"
        super().__init__()
        self.hw = hw

        self.cls2idx = {}
        self.__parse_cls_file(cls_file)

        self.img_paths = []
        self.targets = []
        self.__parse_data_file(data_file)
    
    def num_classes(self):
        return len(self.cls2idx)

    def __parse_cls_file(self, cls_file):
        ## cls_file
        #   class_a
        #   class_b
        #   ...
        with open(cls_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        lines = [i.strip() for i in lines] # "\n"
        
        for idx, name in enumerate(lines, 1):
            self.cls2idx[name] = idx
    
    ## data_file
    #   img_path x1,y1,x2,y2,cls_name;x1,y1,x2,y2,cls_name
    def __parse_data_file(self, data_file):
        with open(data_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            boxes_per_img = []
            line = line.strip()
            img_path, boxes = line.split(" ")

            self.img_paths += [img_path]
            boxes = boxes.split(";")
            for box in boxes:
                box = box.split(",")
                x1, y1, x2, y2 = [float(i) for i in box[0:4]]
                cls_name = box[4]
                boxes_per_img.append([x1, y1, x2, y2, self.cls2idx[cls_name]])
            self.targets.append(boxes_per_img)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_pil = Image.open(self.img_paths[index])
        w, h = img_pil.size
        img_pil = img_pil.resize(self.hw)

        # img_tensor = ToTensor()(img_pil)
        img_tensor = img_pil
        img_tensor = (img_tensor-0.5) / 0.5
        boxes_tensor = torch.Tensor(self.targets[index])
        boxes_tensor[:, 0] /= w
        boxes_tensor[:, 1] /= h
        boxes_tensor[:, 2] /= w
        boxes_tensor[:, 3] /= h
        return {"img": img_tensor, "boxes":boxes_tensor}


def collate_fn(batch_data):
    batch_size = len(batch_data)
    imgs_tensor = torch.stack([i["img"] for i in batch_data], dim=0)
    
    num_boxes = [i["boxes"].size(0) for i in batch_data]
    max_num = max(num_boxes)
    boxes_tensor = torch.zeros((batch_size, max_num, 5)) - 1
    for idx, (data, num_box) in enumerate(zip(batch_data, num_boxes)):
        boxes_tensor[idx, 0:num_box, :] = data["boxes"]
    return {"img": imgs_tensor, "boxes":boxes_tensor}
