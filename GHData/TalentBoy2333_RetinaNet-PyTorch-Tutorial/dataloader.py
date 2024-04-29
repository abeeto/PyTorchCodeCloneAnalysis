import sys
import os
import numpy as np 
import cv2

import torch
import torch.utils.data as Data
from torchvision import transforms

from pycocotools.coco import COCO

from utils import collater, Resizer, Augmenter, Normalizer, UnNormalizer


class CocoDataset(Data.Dataset):
    """
    Coco dataset 
    """
    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()
        print('classes:')
        print(self.classes)

    def load_classes(self):
        # load class names (name -> label)
        '''
        categories 是所有类别的dict()组成的list() 
        dict 中的信息为'supercategory', 'id', 'name'
        举个栗子：{'supercategory': 'person', 'id': 1, 'name': 'person'}
        一共有 90 各类别，但是 coco dataset 的 trainvel 样本中只有80个类别 
        '''
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        '''
        self.classes: {'person': 0, ...}，0~79

        由于coco dataset 的 trainvel 样本中只有80个类别，因此，
        训练样本中的 label 并不是按照顺序 0~79 排列的，
        需要用 self.coco_labels 分别记录80个类别在 coco dataset 中的label
        self.coco_labels是一个dict()，key是训练样本中的id，
        value是 id 对应的 coco dataset 中的label

        self.coco_labels_inverse 是 self.coco_labels dict()的反向
        '''
        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        '''
        self.labels 是 self.classes dict()的反向
        '''
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        '''
        image_info example:
        {'license': 4, 'file_name': '000000498265.jpg', 
        'coco_url': 'http://images.cocodataset.org/train2017/000000498265.jpg', 
        'height': 480, 'width': 640, 'date_captured': '2013-11-22 15:45:06', 
        'flickr_url': 'http://farm6.staticflickr.com/5010/5356425076_2fd6854ef2_z.jpg', 
        'id': 498265}
        '''
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = cv2.imread(path)

        if len(img.shape) == 2:
            img = cv2.imread(path, 0)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        '''
        annotations_ids: list() 
        保存了 imgIds 图像中所有 object 的 id
        '''
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        '''
        获取各个 object 的信息，每个 object 的信息是一个dict()，
        各个 object 的信息组成一个list()
        {'segmentation': [[348.4, 287.63, 388.54, 303.0, 420.13, 292.76, 442.34, 
        277.39, 463.69, 238.1, 461.12, 209.07, 453.44, 191.99, 430.38, 174.91, 
        410.74, 173.21, 393.66, 177.48, 362.07, 192.85, 335.6, 218.46, 326.2, 
        250.06, 335.6, 273.97, 347.55, 290.19]], 
        'area': 12924.149249999999, 'iscrowd': 0, 'image_id': 259976, 
        'bbox': [326.2, 173.21, 137.49, 129.79], 'category_id': 60, 'id': 1572371}
        '''
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            # annotation: [x, y, w, h, class_id]
            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80

def imshow_img_anno(image, anno):
    import matplotlib.pyplot as plt   
    unnormalize = UnNormalizer()
    image = 255 * unnormalize(image)
    image = torch.clamp(image, min=0, max=255).data.numpy()
    image = np.transpose(image, (1, 2, 0)).astype(np.uint8)

    anno = anno.data.numpy()
    for x1, y1, x2, y2, c in anno:
        print(x1, y1, x2, y2, c)
        if x1 != -1:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 1)
            # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
            font = cv2.FONT_HERSHEY_SIMPLEX
            image = cv2.putText(image, str(int(c)), (x1, y1), font, 0.5, (0,255,0), 1)
        else:
            break

    image = image.get()
    print(image.shape)
    plt.figure() 
    image = image[:,:,[2,1,0]]
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    transform = transforms.Compose([Normalizer(), Augmenter(), Resizer()]) 
    dataset = CocoDataset('./data/coco/', 'train2017', transform)
    dataset_size = len(dataset)
    print(dataset_size)

    data_loader = Data.DataLoader(dataset, 2, num_workers=2, shuffle=True, \
                                  collate_fn=collater, pin_memory=True)

    for epoch_num in range(2):
        for iter_num, data in enumerate(data_loader):
            print(
                'epoch:', epoch_num, 
                'iter_num:', iter_num
            )
            print('image:', data['img'].size())
            print('annot:', data['annot'].size())
            print('scale:', data['scale'])
            # imshow_img_anno(data['img'][0], data['annot'][0])
            break 
        break