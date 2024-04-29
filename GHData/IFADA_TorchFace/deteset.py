import torch
from torch.autograd import Variable
from PIL import Image, ImageDraw
import numpy as np
from tool import utils
import nets
from torchvision import transforms
import time


class Detector:

    def __init__(self, pnet_param=r'D:\Pyproject\TorchFace\param\pnet.pkl',
                 rnet_param=r'D:\Pyproject\TorchFace\param\rnet.pkl',
                 onet_param=r'D:\Pyproject\TorchFace\param\onet.pkl', isCuda=True):

        self.isCuda = isCuda
        self.pnet = nets.Pnet()
        self.rnet = nets.Rnet()
        self.onet = nets.Onet()

        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()

        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self._img_transform = transforms.Compose([transforms.ToTensor()])

    def deteset(self, image):
        pnet_boxes = self._pnet_deteset(image)
        # print('pbox',pnet_boxes)
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        print(pnet_boxes)
        rnet_boxes = self._rnet_deteset(image, pnet_boxes)
        # print('rbox',rnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])

        print('rnet',rnet_boxes)
        onet_boxes = self._onet_deteset(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])
        print('onet',onet_boxes)
        return pnet_boxes,rnet_boxes,onet_boxes



    def _pnet_deteset(self, image):
        boxes = []
        img = image
        w, h = img.size
        print(w, h)
        min_side = min(w, h)
        scale = 1

        while min_side >= 12:
            img_data = self._img_transform(img)
            if self.isCuda:
                img_data = img_data.cuda()
            # 增加维度(增加批次)
            img_data.unsqueeze_(0)
            # print('imgshape:', img_data.shape)

            _cls, _offset = self.pnet(img_data)  # NCHW
            #  print('置信度shape', _cls.shape, '偏移量', _offset.shape)
            cls = _cls[0][0].cpu().data  # 置信度的c为1所以提取出HW
            # rint('_cls[0][0]:', cls.shape)
            # print('cls', cls)
            offset = _offset[0].cpu().data  # 偏移的c为4所以需要提取CHW
            idxs = torch.nonzero(torch.gt(cls, 0.6))  # 怎么和0.6作对比？索引值是大于0.6排序（0,1）H是y，W是x（0,1）
            # print('idxs', idxs)
            for idx in idxs:
                boxes.append(self.__box(idx, offset, cls[idx[0], idx[1]], scale))  # ?????

            scale *= 0.7
            _w = int(w * scale)
            _h = int(h * scale)
            img = img.resize((_w, _h), Image.ANTIALIAS)
            min_side = min(_w, _h)
        # print('boxes',np.array(boxes))
        return utils.nms(np.array(boxes), 0.5)

    # R Net

    def _rnet_deteset(self, image, pnet_boxes):
        _img_dataset = []
        _pnet_boxes = utils.convert_to_squre(pnet_boxes)
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            _img = image.crop((_x1, _y1, _x2, _y2))
            img = _img.resize((24, 24), Image.ANTIALIAS)
            img_data = self._img_transform(img)
            _img_dataset.append(img_data)
        img_dataset = torch.stack(_img_dataset)

        if self.isCuda:
            img_dataset = img_dataset.cuda()
        _cls, _offset = self.rnet(img_dataset)
        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()
        boxes = []
        indxs, _ = np.where(cls > 0.7)

        for idx in indxs:
            _box = _pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x2
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]
            boxes.append([x1, y1, x2, y2, cls[idx][0]])

        return utils.nms(np.array(boxes), 0.5)

    def _onet_deteset(self, image, rnet_boxes):
        _image_dataset = []
        _rnet_dataset = utils.convert_to_squre(rnet_boxes)
        for _box in _rnet_dataset:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48), Image.ANTIALIAS)

            img_data = self._img_transform(img)
            _image_dataset.append(img_data)

        image_dataset = torch.stack(_image_dataset)#加图片放入image_dataset中，通过stack给图片的数量价格批次

        if self.isCuda:
            image_dataset = image_dataset.cuda()

        _cls, _offset = self.onet(image_dataset)
        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()
        boxes = []
        indxs,_ = np.where(cls > 0.97)
        for idx in indxs:
            _box = _rnet_dataset[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            boxes.append([x1, y1, x2, y2, cls[idx][0]])

        return utils.nms(np.array(boxes), 0.7, isMin=True)

    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):

        start_index_ = np.array(start_index, np.float64)
        _x1 = (start_index_[1] * stride) / scale  # 反算回去步长*索引
        _y1 = (start_index_[0] * stride) / scale
        _x2 = (start_index_[1] * stride + side_len) / scale
        _y2 = (start_index_[0] * stride + side_len) / scale

        ow = _x2 - _x1
        oh = _y2 - _y1

        _offset = offset[:, start_index[0], start_index[1]]  # ？？？？？？

        x1 = _x1 + _offset[0] * ow
        y1 = _y1 + _offset[1] * oh
        x2 = _x2 + _offset[2] * ow
        y2 = _y2 + _offset[3] * oh

        return [x1, y1, x2, y2, cls]


if __name__ == '__main__':

    image_file = r"D:\Pyproject\TorchFace\Test\timg.jpg"
    detector = Detector()

    with Image.open(image_file) as im:

        p_box,r_box,o_box = detector.deteset(im)
        imDraw = ImageDraw.Draw(im)
        for box in o_box:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            imDraw.rectangle((x1, y1, x2, y2), outline='red')
            imDraw.text((box[0], box[1]), str(box[4]), fill='black')
        im.show()
