import os
import json
configs = json.load(open("config.json"))
os.chdir(configs['working_dir'])
import numpy as np
import torch
from PIL import Image
import json
import cv2
import skimage.draw
class Dataset(object):
    def __init__(self, root,  transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "img"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "ann"))))
        self.class_map = configs["class_map"]


    def __getitem__(self, idx):
        # load images ad masks
        
        img_path = os.path.join(self.root, "img", self.imgs[idx])
        ann_path = os.path.join(self.root, "ann", self.annotations[idx])
        #print(f"-----loding image {img_path}-----")
        #print(img_path)

        img = Image.open(img_path).convert("RGB")
        anns = json.load(open(ann_path))

        tags = [obj['classTitle'] for obj in anns['objects']]
       
        labels = [self.class_map[a] for a in tags]
        if configs["model_name"] == "mask_rcnn":

            polygons = [obj['points'] for obj in anns['objects']]

            
            image_height = anns["size"]["height"]
            image_width = anns["size"]["width"]

            
            masks = np.zeros([len(polygons), image_height, image_width], dtype=np.uint8)
            for i, p in enumerate(polygons):
                pts = p['exterior']
                all_x = [int(p[0]) for p in pts]
                all_y = [int(p[1]) for p in pts]
                rr, cc = skimage.draw.polygon(all_y, all_x)
                masks[i, rr, cc] = 1

            num_objs = len(polygons)
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])

                
                boxes.append([xmin, ymin, xmax, ymax])
            
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor([idx])
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            # suppose all instances are not crowd
            target = {}
            target["area"] = area
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["masks"] = masks
            target["iscrowd"] = iscrowd

            

            img = self.transforms(img)


            #print(target)
            return img, target
        elif configs["model_name"] == "faster_rcnn": 

            bbox = [obj['points'] for obj in anns['objects']]

            bb_box=[]
            for i, b in enumerate(bbox):
                bb=b["exterior"]
                x_min,y_min,x_max,y_max=bb[0][0],bb[0][1],bb[1][0],bb[1][1]
                tmp=0
                if x_min > x_max:
                    tmp = x_max
                    x_max = x_min
                    x_min = tmp
                if y_min > y_max:
                    tmp = y_max
                    y_max = y_min
                    y_min = tmp
                    
                bb_box.append([x_min,y_min,x_max,y_max])
            num_objs=len(bbox)
            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(bb_box, dtype=torch.float32)
            # there is only one class
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target


    def __len__(self):
        return len(self.imgs)


#if __name__ == '__main__':
    #dataset = Dataset('./temp_dataset/train', None)
    

    #img, target = dataset.__getitem__(0)
    #print(target)
    #print('image shape', np.array(img).shape)
    
    #mask = Image.open('./temp.png')
    
    #mask = np.array(mask)
    #print(mask.shape)
#     # instances are encoded as different colors
#     obj_ids = np.unique(mask)
    
#     # first id is the background, so remove it
#     obj_ids = obj_ids[1:]
#     #print(obj_ids)

#     # split the color-encoded mask into a set
#     # of binary masks
#     print(mask.shape[0])
    
    


    