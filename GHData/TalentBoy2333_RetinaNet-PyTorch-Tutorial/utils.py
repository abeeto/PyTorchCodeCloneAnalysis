import random
import numpy as np  
import cv2 

import torch
import torch.nn as nn
cuda = torch.cuda.is_available()

def collater(data):
    '''
    创建dataloader时，用于提供torch.utils.Data.DataLoader()函数的参数collate_fn
    作用：同一批次(batch)中不同大小的图像，将其调整为相同大小
    '''
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    # 获取各图像最大的宽度和高度
    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    # 创建固定大小的图像框架
    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    # 将各图像送入图像框架padded_imgs，未填满的部分置0
    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    # 获取一批图像中，包含目标个数最多的图像中的目标个数
    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:
        # 创建固定大小的图像标注框架
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        if max_num_annots > 0:
            # 将各图像的标注送入图像标注框架annot_padded中，未填满的部分置-1
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    '''
    # PyTorch中图像输入的格式为(batch, channel, width, height)
    # 与我们得到的数据格式不同，需要进行更改
    # (batch, width, height, channel) -> (batch, channel, width, height)
    '''
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """
    Convert ndarrays in sample to Tensors.
    调整图像的尺度，将图像的大小维持在一个范围内
    """
    def __call__(self, sample, min_side=400, max_side=800):
        image, annots = sample['img'], sample['annot']
        rows, cols, cns = image.shape

        '''
        首先，将图像的短边调整为 min_side，长边做相应调整，
        如果，调整之后，长边的长度大于了 max_side，
        那么，就将长边调整为 max_side，短边也做相应调整
        '''
        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        # image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        image = cv2.resize(image, (int(round(cols*scale)), int(round(rows*scale))))
        rows, cols, cns = image.shape

        '''
        由于 ResNet+FPN 的神经网络结构会对图像进行下采样，最大的下采样倍数为32，
        因此，输入图像的大小最好为32的倍数，
        为此，我们定义pad_w和pad_h，使用0值将图像大小补为32的倍数
        '''
        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        # 调整图像后，还需要对标注(bounding box)进行调整
        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """
    Convert ndarrays in sample to Tensors.
    样本扩增，方法为随机翻转图像
    """
    def __call__(self, sample, flip_x=0.5):
        # 产生随机数来进行随机操作
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            # 左右翻转图像
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            x_tmp = x1.copy()

            # 左右翻转图像中各个目标的 bounding box
            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):
    '''
    对图像灰度值进行正则化，在该正则化之前，图像灰度值已经经过了 ' / 255' 的处理
    这里是，将图像灰度值正则化为高斯分布
    '''
    def __init__(self):
        # 先验
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        image, annots = sample['img'], sample['annot']
        return {'img':((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


class UnNormalizer(object):
    '''
    class Normalizer(object) 的反向操作
    '''
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) have be normalized.
        Returns:
            Tensor: UnNormalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def cal_iou(anchors, boxs):
    '''
    计算一张图像产生的所有anchor和该图annotations中的bounding box之间的IoU
    :param anchors: [-1, 4], [x1, y1, x2, y2]
    :param boxs: annotations [-1, 4], [x1, y1, x2, y2]
    '''
    anchors_w = anchors[:, 2] - anchors[:, 0] # [n, ]
    anchors_h = anchors[:, 3] - anchors[:, 1] # [n, ]
    boxs_w = boxs[:, 2] - boxs[:, 0] # [m, ]
    boxs_h = boxs[:, 3] - boxs[:, 1] # [m, ]

    anchors_area = anchors_w * anchors_h
    boxs_area = boxs_w * boxs_h

    '''
    # torch.unsqueeze(anchors[:, 0], dim=1) 
    # 将anchors的x1的shape进行转变, [n, ] -> [n, 1]
    # boxs的x1: [m, ]
    # 使用torch.max()函数，对不同维度的向量进行比较，可以得到一一对应的对比结果
    # overlap_x1: [n, m]
    '''
    overlap_x1 = torch.max(torch.unsqueeze(anchors[:, 0], dim=1), boxs[:, 0])
    overlap_y1 = torch.max(torch.unsqueeze(anchors[:, 1], dim=1), boxs[:, 1])
    overlap_x2 = torch.min(torch.unsqueeze(anchors[:, 2], dim=1), boxs[:, 2])
    overlap_y2 = torch.min(torch.unsqueeze(anchors[:, 3], dim=1), boxs[:, 3])

    '''
    overlap_w和overlap_h可能会出现负数的情况，表示anchor和anno没有交集
    这时需要将其w和h置为0，因为之后算交集面积的时候，负数会带来困扰
    '''
    overlap_w = torch.clamp(overlap_x2 - overlap_x1, min=0) 
    overlap_h = torch.clamp(overlap_y2 - overlap_y1, min=0)
    overlap_area = overlap_w * overlap_h 
    
    # torch.unsqueeze(anchors_area, dim=1) + boxs_area
    # 同理，将anchors_area进行转变 [n, ] -> [n, 1] 与[m, ] 的boxs_area相加
    # 得到[n, m]
    collection_area = torch.unsqueeze(anchors_area, dim=1) + boxs_area - overlap_area
    # 防止最后计算出来的IoU过小而导致log()函数值输出'nan'
    collection_area = torch.clamp(collection_area, min=1e-8)

    iou = overlap_area / collection_area

    return iou

class Decoder(nn.Module):
    '''
    将 RetinaNet 输出的 classification, localization 转化为 box 的形式：
    [x1, y1, x2, y2]
    '''
    def __init__(self):
        super(Decoder, self).__init__() 
        '''
        计算focal_loss时，我们进行过如下操作：
        if cuda:
            d_stack = d_stack / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
        else:
            d_stack = d_stack / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])
        这里需要进行反操作
        '''
        if cuda:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
        else:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))

        if cuda:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
        else:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))

    def forward(self, anchors, d_stack):
        anchors_w  = anchors[:, 2] - anchors[:, 0]
        anchors_h = anchors[:, 3] - anchors[:, 1]
        anchors_cx   = anchors[:, 0] + 0.5 * anchors_w
        anchors_cy   = anchors[:, 1] + 0.5 * anchors_h

        dx = d_stack[0, :, 0] * self.std[0] + self.mean[0]
        dy = d_stack[0, :, 1] * self.std[1] + self.mean[1]
        dw = d_stack[0, :, 2] * self.std[2] + self.mean[2]
        dh = d_stack[0, :, 3] * self.std[3] + self.mean[3]

        pred_w = torch.exp(dw) * anchors_w
        pred_h = torch.exp(dh) * anchors_h
        pred_cx = dx * anchors_w + anchors_cx
        pred_cy = dy * anchors_h + anchors_cy

        pred_x1 = pred_cx - 0.5 * pred_w
        pred_y1 = pred_cy - 0.5 * pred_h
        pred_x2 = pred_cx + 0.5 * pred_w
        pred_y2 = pred_cy + 0.5 * pred_h

        pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)

        return pred_boxes



if __name__ == '__main__': 
    print('..')