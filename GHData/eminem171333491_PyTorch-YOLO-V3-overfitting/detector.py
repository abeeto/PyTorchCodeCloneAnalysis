from model import *
import cfg
import torch
import numpy as np
import PIL.Image as pimg
import PIL.ImageDraw as draw
import tool
from PIL import Image,ImageFont
from torchvision import transforms

classname = {0:"人类",1:"大熊猫",2:"小熊猫",3:"浣熊"}

class Detector(torch.nn.Module):

    def __init__(self,save_path):
        super(Detector, self).__init__()
        self.net = MainNet().cuda()
        self.net.load_state_dict(torch.load(save_path))
        self.net.eval()

    def forward(self, input, thresh, anchors):
        output_13, output_26, output_52 = self.net(input)
        idxs_13, vecs_13 = self._filter(output_13, thresh)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])
        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])
        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])
        print(boxes_13.shape)
        print(boxes_26.shape)
        print(boxes_52.shape)
        return torch.cat([boxes_13, boxes_26, boxes_52], dim=0)

    def _filter(self, output, thresh):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        mask = torch.sigmoid(output[..., 0]) > thresh
        idxs = mask.nonzero()
        vecs = output[mask]
        # print(vecs)
        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors):
        if len(idxs) == 0:
            return torch.randn(0, 6).cuda()
        else:
            anchors = torch.tensor(anchors, dtype=torch.float32).cuda()
            a = idxs[:, 3]  # 建议框:3

            # confidence = vecs[:, 0]
            # "压缩置信度值到0-1之间"
            confidence = torch.sigmoid(vecs[:, 0])
            # confidence = vecs[:, 0]
            _classify = vecs[:, 5:]
            classify = torch.argmax(_classify, dim=1).float()

            cy = (idxs[:, 1].float() + torch.sigmoid(vecs[:, 2])) * t
            cx = (idxs[:, 2].float() + torch.sigmoid(vecs[:, 1])) * t
            # cy = (idxs[:, 1].float() + vecs[:, 2]) * t
            # cx = (idxs[:, 2].float() + vecs[:, 1]) * t
            w = anchors[a, 0] * torch.exp(vecs[:, 3])
            h = anchors[a, 1] * torch.exp(vecs[:, 4])
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = x1 + w
            y2 = y1 + h
            # print(confidence)
            out = torch.stack([confidence, x1, y1, x2, y2, classify], dim=1)
            return out

if __name__ == '__main__':
    save_path = "models/net_yolo5.pth"
    detector = Detector(save_path)
    # tf = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229, 0.224, 0.225])]
    # )
    # y = detector(torch.randn(3, 3, 416, 416), 0.3, cfg.ANCHORS_GROUP)
    # print(y.shape)
    font_size = 25
    font = ImageFont.truetype("simhei.ttf",font_size,encoding="utf-8")

    img1 = pimg.open(r'data\image\11.jpg')
    im1 = img1.convert('RGB')
    w, h = im1.size
    background = Image.new('RGB', size=(max(w, h), max(w, h)), color=(0, 0, 0))  # 创建背景图
    length = int(abs(w - h) // 2)  # 一侧需要填充的长度
    box = (length, 0) if w < h else (0, length)  # 粘贴的位置
    background.paste(im1, box)
    im = background.resize((416, 416))
    # a = np.zeros([max(im1.size[0], im1.size[1]), max(im1.size[0], im1.size[1]), 3])  # 以最大边长生成0矩阵
    # img_zero = pimg.fromarray(np.uint8(a))  # 0矩阵转为PIL
    # img_zero.paste(im1, (0, 0, im1.size[0], im1.size[1]))  # 将原来的图片贴到0矩阵生成的图片上
    # img = img_zero.resize((416, 416), pimg.ANTIALIAS)
    img = np.array(im) / 255
    img = torch.Tensor(img)
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)
    img = img.cuda()

    out_value = detector(img, 0.6, cfg.ANCHORS_GROUP)
    boxes = []

    for j in range(4):
        classify_mask = (out_value[..., -1] == j)
        _boxes = out_value[classify_mask]
        boxes.append(tool.nms(_boxes.cpu()))

    for boxs in boxes:
        for box in boxs:
            try:
                img_draw = draw.ImageDraw(img1)
                # confidence = round(box[0].item(),2)

                c,x1, y1, x2, y2 ,cls = box[0:6]

                print(c.item(),x1.item(), y1.item(), x2.item(), y2.item())

                x_1 = min(max(x1, 0), 416)  # 在416的图像框上的坐标
                y_1 = min(max(y1, 0), 416)
                x_2 = min(max(x2, 0), 416)
                y_2 = min(max(y2, 0), 416)
                # print(x_1, y_1, x_2, y_2)

                xx_1 = x_1 * max(im1.size[0], im1.size[1]) / 416  # 还原坐标
                yy_1 = y_1 * max(im1.size[0], im1.size[1]) / 416
                xx_2 = x_2 * max(im1.size[0], im1.size[1]) / 416
                yy_2 = y_2 * max(im1.size[0], im1.size[1]) / 416

                if im1.size[0] >= im1.size[1]:  # 原图的长大于宽
                    xx1 = min(max(xx_1, 0), im1.size[0])
                    yy1 = min(max(yy_1, 0), im1.size[1])
                    xx2 = min(max(xx_2, 0), im1.size[0])
                    yy2 = min(max(yy_2, 0), im1.size[1])
                else:  # 原图的长小于宽
                    xx1 = min(max(xx_1, 0), im1.size[0])
                    yy1 = min(max(yy_1, 0), im1.size[1])
                    xx2 = min(max(xx_2, 0), im1.size[0])
                    yy2 = min(max(yy_2, 0), im1.size[1])

                # print(x1.item(), y1.item(), x2.item(), y2.item())
                img_draw.rectangle((xx1, yy1, xx2, yy2),outline=(235,61,131),width=4)
                img_draw.text((xx1,yy1-font_size),classname[int(cls)],fill=(235,61,131),font=font)
            except:
                continue
    img1.show()
