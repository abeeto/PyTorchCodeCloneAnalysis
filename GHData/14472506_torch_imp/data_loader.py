import torch
import torchvision.transforms as T
import torch.utils.data as data
import os 
from pycocotools.coco import COCO
from PIL import Image
import numpy as np

class COCOLoader(data.Dataset):
    """
    Custom coco data loader form pytorch
    """
    def __init__(self, root, json, image_transforms=None, target_transforms=None):
        """
        Details of init
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.imgs.keys())
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms

    def __getitem__(self, idx):
        """
        Detail Get item
        """
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds = img_id)
        
        # generating target
        anns = self.coco.loadAnns(ann_ids)

        labels = []
        boxes = []        
        masks_list = []
        areas = []
        iscrowds = []
        
        for ann in anns:
            
            labels.append(ann['category_id'])
            areas.append(ann['area'])

            bbox = ann['bbox']            
            new_bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            boxes.append(new_bbox)
    
            if ann["iscrowd"]:
                iscrowds.append(1)
            else:
                iscrowds.append(0)

            mask = self.coco.annToMask(ann)
            mask == ann['category_id']
            masks_list.append(torch.from_numpy(mask))

            #pos = np.where(mask == True)
            #xmin = np.min(pos[1])
            #xmax = np.max(pos[1])
            #ymin = np.min(pos[0])
            #ymax = np.max(pos[0])
            #boxes.append([xmin, ymin, xmax, ymax])            

        # to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(areas, dtype=torch.int64)
        masks = torch.stack(masks_list, 0)
        iscrowd = torch.as_tensor(iscrowds, dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # loading image
        image_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, image_path)).convert('RGB')
        im_conv = T.ToTensor()
        img = im_conv(img)

        # applying transforms
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return img, target

    def __len__(self):
        return len(self.ids)
        
#if __name__ == "__main__":
#    
#    root_dir = "data/jersey_royal_ds/train"
#    json_root = "data/jersey_royal_ds/train/train.json"
#
#    loader = COCOLoader(root_dir, json_root)
#
#    img, target = loader.__getitem__(114)
#
#    print(target)
