import torch 
import torch.nn as nn 
import numpy as np 

'''
引用 'Focal Loss for Dense Object Detection' 论文原文：
'The anchors have areas of 32^2 to 512^2 on pyramid levels P3 to P7, respectively. 
As in [20], at each pyramid level we use anchors at three aspect ratios 
{1:2, 1:1, 2:1}. For denser scale coverage than in [20], at each level we add 
anchors of sizes {2^0, 2^(1/3), 2^(2/3)} of the original set of 3 aspect ratio 
anchors. This improve AP in our setting. In total there are A = 9 anchors per 
level and across levels they cover the scale range 32 - 813 pixels with respect 
to the network’s input image.'
'''

class Anchor(nn.Module):
    '''
    生成一张图像的所有anchor, [x, y, w, h]
    x: anchor中心点横坐标
    y: anchor中心点纵坐标
    w: anchor宽
    h: anchor高
    '''
    def __init__(self):
        super(Anchor, self).__init__() 
        # P3~P7特征图大小分别是原图像大小的
        # [1/2**3, 1/2**4, 1/2**5, 1/2**6, 1/2**7]
        self.pyramid = [3,4,5,6,7] 
        self.anchor_sizes = [32, 64, 128, 256, 512]
        self.strides = [2**i for i in self.pyramid] # [8, 16, 32, 64, 128]
        self.ratios = [1/2, 1, 2]
        self.scales = [2**0, 2**(1/3), 2**(2/3)]
        
    def create_anchor(self, anchor_size, ratios, scales):
        '''
        :在一个anchor_size的基础上创建若干不同形状大小的anchor，一个点上的所有anchor
        :param anchor_size: int
        :param ratios: list()
        :param scales: list()
        :return anchors: np.array() shape: [-1, 4], [0, 0, w, h]
        '''
        anchor_num = len(ratios) * len(scales)
        anchors = np.zeros((anchor_num, 4))

        # 首先定义不同尺度的anchor，尺寸分别为anchor_size的{2^0, 2^(1/3), 2^(2/3)}倍
        '''
        举个例子说明'np.repeat()'函数的功能：
        ratios = [1/2, 1, 2]
        scales = [2**0, 2**(1/3), 2**(2/3)]
        np.repeat(scales, len(ratios))
        output: 
        [1.         1.         1.         1.25992105 1.25992105 1.25992105
         1.58740105 1.58740105 1.58740105]
        '''
        anchors[:, 2] = anchor_size * np.repeat(scales, len(ratios))
        anchors[:, 3] = anchor_size * np.repeat(scales, len(ratios))
        # print(anchors)

        # 将各个尺度的正方形anchor进行变形，面积不变，形状变为长宽比为{1:2, 1:1, 2:1}的矩形
        '''
        'np.repeat(input, shape)' 将input视为一个单元，创建一个shape形状的矩阵，
        矩阵中的每个元素都是这个单元
        举个栗子：
        ratios = [1/2, 1, 2]
        scales = [2**0, 2**(1/3), 2**(2/3)]
        np.tile(ratios, len(scales))
        输出为：
        [0.5 1.  2.  0.5 1.  2.  0.5 1.  2. ]
        '''
        anchors[:, 2] = anchors[:, 2] * np.sqrt(np.tile(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 3] / np.sqrt(np.tile(ratios, len(scales)))

        return anchors 

    def imshow_anchors(self):
        '''
        anchors展示
        使用方法：
        anchor = Anchor()
        anchor.imshow_anchors()
        '''
        anchors = self.create_anchor(64, self.ratios, self.scales)
        import cv2 
        import matplotlib.pyplot as plt  
        image = np.ones((200, 200)) * 255
        for _, _, w, h in anchors:
            x1, y1 = int(100 - w / 2), int(100 - h / 2)
            x2, y2 = int(100 + w / 2), int(100 + h / 2)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 1)
        image = image.astype(np.uint8)
        plt.figure() 
        plt.imshow(image)
        plt.show()

    def create_grid(self, feat_shape, stride):
        '''
        :在原图像上创建网格，得到各个网格的中心坐标
        :param feat_shape: [h, w] of feature map
        :param stride: int
        :return coor_xs: 各个网格中心点的横坐标, list()
        :return coor_ys: 各个网格中心点的纵坐标, list()
        '''

        '''
        这里我踩坑了，因为整个代码里都是[x, y, w, h]这样的格式，所以我惯性的认为
        feat_shape肯定也是同样的格式，其实不然，feat_shape是使用 torch 里的
        .size()函数得到的，其返回的格式为[h, w]，需要及其注意，debug了1个多小时。。
        '''
        h, w = feat_shape.astype(np.int) # !!!
        xs = (np.arange(0, w) + 0.5) * stride
        ys = (np.arange(0, h) + 0.5) * stride

        # 在原图像上创建image_shape大小的网格图
        # coor_xs, coor_ys 分别代表各网格中心的 横/纵 坐标
        coor_xs, coor_ys = np.meshgrid(xs, ys)

        # 将二维网格转换为一维向量
        coor_xs = coor_xs.flatten()
        coor_ys = coor_ys.flatten()
        return coor_xs, coor_ys
        
    def grid_anchors(self, grid, anchors):
        '''
        :合并grid和anchors，得到一个尺度特征图上的所有anchor
        :param grid: [coor_xs, coor_ys]
        :param anchors: np.array() shape: [-1, 4], [0, 0, w, h]
        :return anchors_level: np.array() shape: [-1, 4]
        '''
        coor_xs, coor_ys = grid 
        grid_num = coor_xs.shape[0]
        anchor_num = anchors.shape[0]
        # 改变anchors数据格式, [0, 0, w, h] -> [x1, y1, x2, y2]
        anchors[:, 0:2] -= anchors[:, 2:4] / 2
        anchors[:, 2:4] = anchors[:, 2:4] / 2
        # print(anchors)

        # 获得网格中心点坐标 np.array() shape: [gird_num, 4], [x, y, x, y]
        centers = np.vstack((coor_xs, coor_ys, coor_xs, coor_ys)).transpose()
        # print(centers)
        '''
        np.array()相加:
        shape(100,1,4) + shape(1,9,4) = shape(100,9,4)

        其中，[:, :, 0:4] 
        配合 anchors 的 [0, 0, w, h] 和 centers 的 [x, y, 0, 0]
        可以合成 [x, y, w, h]
        '''
        centers = centers.reshape((1, grid_num, 4)) # (1,-1,4)
        anchors = anchors.reshape((1, anchor_num, 4)).transpose((1, 0, 2)) # (-1,1,4)
        anchors_level = centers + anchors 
        anchors_level = anchors_level.reshape((-1, 4))
        return anchors_level

    def forward(self, images):
        '''
        :param images: source image 
        :return total_anchors: all anchors [-1, 4], [x, y, w, h]
        '''
        image_shape = images.size()[2:]
        image_shape = np.array(image_shape)
        # np.ceil() 向上取整
        feat_shapes = [np.ceil(image_shape / 2**i) for i in self.pyramid]

        total_anchors = np.zeros((0, 4))
        for i, feat_shape in enumerate(feat_shapes):
            anchor_size = self.anchor_sizes[i]
            stride = self.strides[i]
            # print(anchor_size, stride)
            anchors = self.create_anchor(anchor_size, self.ratios, self.scales)
            grid = self.create_grid(feat_shape, stride)
            anchors_level = self.grid_anchors(grid, anchors)
            # print(anchors_level.shape)
            total_anchors = np.append(total_anchors, anchors_level, axis=0)
            # break

        total_anchors = torch.Tensor(total_anchors)
        return total_anchors



if __name__ == '__main__':
    fake_images = np.random.uniform(size=[1, 3, 640, 640])
    fake_images = torch.Tensor(fake_images)
    print(fake_images.size())

    anchor = Anchor()
    # anchor.imshow_anchors()
    total_anchors = anchor(fake_images)
    print(total_anchors.shape)