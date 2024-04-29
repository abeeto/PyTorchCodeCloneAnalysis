import time
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import torch 
from torchvision.ops import nms

from retina_net import RetinaNet 
from anchor import Anchor
from utils import Decoder

cuda = torch.cuda.is_available()


def get_RetinaNet(param_path='./param/param_epoch100.pkl'):
    '''
    获取 RetinaNet 模型
    '''
    retinanet = RetinaNet(training=False) 
    retinanet.eval()
    if cuda:
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    if param_path:
        print('Loading parameters of model.')
        retinanet.load_state_dict(torch.load(param_path))
    return retinanet 

def imshow_result(image, bounding_boxes):
    '''
    显示结果
    '''
    nms_scores, nms_cls, nms_boxes = bounding_boxes
    for score, cls_ind, box in zip(nms_scores, nms_cls, nms_boxes): 
        x1, y1, x2, y2 = box 
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  
        score = round(float(score), 4)
        cls_ind = int(cls_ind)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        info = str(cls_ind) + '(' + str(score) + ')'
        image = cv2.putText(image, str(info), (x1, y1), font, 0.5, (0,255,0), 1)
    
    # image = image.get()
    plt.figure() 
    image = image[:,:,[2,1,0]]
    plt.imshow(image)
    plt.show()

def predict_one_image(retinanet, image):
    # 对图像进行归一化
    image = image.astype(np.float32)/255.0
    print('source image shape:', image.shape)

    # 将图像调整为适合输入的尺寸，具体细节解释可以查阅'utils.py'的'class Resizer(object)'
    min_side, max_side = 400, 800
    rows, cols, cns = image.shape
    smallest_side = min(rows, cols)
    scale = min_side / smallest_side
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    image = cv2.resize(image, (int(round(cols*scale)), int(round(rows*scale))))
    print('resize image shape:', image.shape)
    rows, cols, cns = image.shape
    pad_w = 32 - rows%32
    pad_h = 32 - cols%32
    net_input = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    net_input[:rows, :cols, :] = image.astype(np.float32)

    # 将 net_input 调整为可以输入 RetinaNet 的格式
    net_input = torch.Tensor(net_input)
    net_input = net_input.unsqueeze(dim=0)
    net_input = net_input.permute(0, 3, 1, 2)
    print('RetinaNet input size:', net_input.size()) 

    anchor = Anchor() 
    decoder = Decoder() 

    if cuda:
        net_input = net_input.cuda() 
        anchor = anchor.cuda()
        decoder = decoder.cuda()

    total_anchors = anchor(net_input)
    print('create anchor number:', total_anchors.size()[0])
    classification, localization = retinanet(net_input)

    pred_boxes = decoder(total_anchors, localization)

    # pred_boxes中的边框，有可能会出现在图像边界以外，需要将其拉回
    height, width, _ = image.shape
    pred_boxes[:, 0] = torch.clamp(pred_boxes[:, 0], min=0)
    pred_boxes[:, 1] = torch.clamp(pred_boxes[:, 1], min=0)
    pred_boxes[:, 2] = torch.clamp(pred_boxes[:, 2], max=width)
    pred_boxes[:, 3] = torch.clamp(pred_boxes[:, 3], max=height)

    # classification: [1, -1, 80]
    # torch.max(classification, dim=2, keepdim=True): [(1, -1, 1), (1, -1, 1)]
    # scores: [1, -1, 1]， 所有anchor对应的置信度最大的类别id
    scores, ss = torch.max(classification, dim=2, keepdim=True)

    scores_over_thresh = (scores > 0.05)[0, :, 0] # [True or False]
    if scores_over_thresh.sum() == 0:
        # no boxes to NMS, just return
        nms_scores = torch.zeros(0)
        nms_cls = torch.zeros(0)
        nms_boxes = torch.zeros(0, 4)
    else:
        # 提取最大置信度超过阈值的 anchor 的 classification
        classification = classification[:, scores_over_thresh, :]
        # 提取最大置信度超过阈值的 anchor 的 pred_boxes
        pred_boxes = pred_boxes[scores_over_thresh, :]
        # 提取最大置信度超过阈值的 anchor 的 scores
        scores = scores[:, scores_over_thresh, :]

        nms_ind = nms(pred_boxes[:,:], scores[0,:,0], 0.5)

        nms_scores, nms_cls = classification[0, nms_ind, :].max(dim=1)
        nms_boxes = pred_boxes[nms_ind, :]

    print('Predict bounding boxes number:', nms_scores.size()[0])
    bounding_boxes = [nms_scores, nms_cls, nms_boxes]

    imshow_result(image, bounding_boxes)



if __name__ == '__main__':
    retinanet = get_RetinaNet(None)
    image = cv2.imread('./data/example.jpg')
    
    predict_one_image(retinanet, image)