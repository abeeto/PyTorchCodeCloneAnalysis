import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms
from tqdm import tqdm
import numpy as np
from chainercv.evaluations import eval_detection_voc
import cv2
import os

from dataset import WSDDN_Dataset
from model import WSDDN

CLASS2ID = {
        "aeroplane": 0,
        "bicycle": 1,
        "bird": 2,
        "boat": 3,
        "bottle": 4,
        "bus": 5,
        "car": 6,
        "cat": 7,
        "chair": 8,
        "cow": 9,
        "diningtable": 10,
        "dog": 11,
        "horse": 12,
        "motorbike": 13,
        "person": 14,
        "pottedplant": 15,
        "sheep": 16,
        "sofa": 17,
        "train": 18,
        "tvmonitor": 19,
    }


def get_args_val():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--base_net", type=str, default="vgg", help="Base network to use"
    )
    parser.add_argument("--state_path", default="/home/rosetta/wsddn.pytorch/weight/pre\.pt", help="Path of trained model's state")
    parser.add_argument("--draw_box", type=bool, default=True, help="draw boxes and save")
    parser.add_argument("--img_dir",  default="/home/rosetta/wsddn.pytorch/data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages")
    parser.add_argument("--result_dir",  default="/home/rosetta/wsddn.pytorch/result")
    args = parser.parse_args()
    return args


def draw_rect(img_id, img, rect, labels, scores, save_dir):

    for i, element in enumerate((rect, labels, scores)):
        s = scores[i]
        #置信度高于0.05再画框
        if(s>0.05):
            r = rect[i]
            topleft = (int(r[0]), int(r[1]))
            lowright = (int(r[2]), int(r[3]))

            l = labels[i]
            l_text = list(CLASS2ID.keys())[list(CLASS2ID.values()).index(l)]

            result_img = cv2.rectangle(img, topleft, lowright, (0, 255, 0), 2)
            result_img = cv2.putText(img, '{} {:.3f}'.format(l_text, s), topleft, cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                     (0, 255, 255), 2)
            cv2.imwrite(os.path.join(save_dir, img_id + '_result.png'), result_img)




def np2gpu(arr, device):
    """Creates torch array from numpy one."""
    arr = np.expand_dims(arr, axis=0)
    return torch.from_numpy(arr).to(device)


def evaluate(args, net, dataloader):
    """Evaluates network."""
    with torch.no_grad():
        net.eval()

        total_pred_boxes = []
        total_pred_scores = []
        total_pred_labels = []
        total_gt_boxes = []
        total_gt_labels = []

        for (
            img_id,
            img,
            boxes,
            scaled_imgs,
            scaled_boxes,
            scores,
            gt_boxes,
            gt_labels,
        ) in tqdm(dataloader, "Evaluation"):

            combined_scores = torch.zeros(len(boxes), 20, dtype=torch.float32)
            batch_scores = np2gpu(scores.numpy(), DEVICE)

            for i, scaled_img in enumerate(scaled_imgs):
                scaled_img = scaled_img.numpy()
                tmp_scaled_boxes = scaled_boxes[i].numpy()

                batch_imgs = np2gpu(scaled_img, DEVICE)
                batch_boxes = np2gpu(tmp_scaled_boxes, DEVICE)

                tmp_combined_scores = net(batch_imgs, batch_boxes, batch_scores)
                # print(tmp_combined_scores)
                combined_scores += tmp_combined_scores.cpu()

            combined_scores /= 10   #在测试集的数据增强中1张图片对应10张处理后的图片，10个为一组得到最终输出

            gt_boxes = gt_boxes.numpy()
            gt_labels = gt_labels.numpy()

            batch_gt_boxes = np2gpu(gt_boxes, DEVICE)
            batch_gt_labels = np2gpu(gt_labels, DEVICE)

            batch_pred_boxes = []
            batch_pred_scores = []
            batch_pred_labels = []


            # 对20个类，每一类画框
            for i in range(20):

                region_scores = combined_scores[:, i]   #以第一张图片为例，shape:torch.Size([2186])

                # 布尔索引  用布尔索引总是会返回一份新创建的数据，原本的数据不会被改变
                # 滤除region_scores<0
                score_mask = region_scores > 0
                #以第一张图片为例，shape:torch.Size([2186])
                #example:tensor([True, True, True,  ..., True, True, True])
                selected_scores = region_scores[score_mask] #shape:torch.Size([2186]),以第一张图片为例，所有区域均保留
                selected_boxes = boxes[score_mask]  #torch.Size([2186, 4])

                # 对于不同的i，即对于不同的类，nms_mask结果不同
                # int64型的 tensor ，包含经过nms处理后的框在selected_boxes中的索引
                # example：
                # tensor([18, 376, 141, ... , 62]),score值高的框的索引排在前面
                nms_mask = nms(selected_boxes, selected_scores, 0.4)


                draw_boxes = selected_boxes[nms_mask].cpu().numpy()
                draw_scores = selected_scores[nms_mask].cpu().numpy()
                draw_labels = np.full(len(nms_mask), i, dtype=np.int32)

                if(args.draw_box):
                    # 不是第一个类且已有结果图片，则在结果图片上画框
                    if((i!=0) and (os.path.exists(os.path.join(args.result_dir, img_id + '_result.png')))):
                        img_for_draw = cv2.imread(os.path.join(args.result_dir, img_id + '_result.png'))
                    else:
                        # 第一个类或仍无结果出现，则打开测试集图片
                        img_for_draw = cv2.imread(os.path.join(args.img_dir, img_id + '.jpg'))

                    draw_rect(img_id, img_for_draw, draw_boxes,
                              draw_labels, draw_scores, args.result_dir)

                batch_pred_boxes.append(draw_boxes)
                batch_pred_scores.append(draw_scores)
                batch_pred_labels.append(draw_labels)

            tmp_pred_boxes = np.concatenate(batch_pred_boxes, axis=0)
            tmp_pred_scores = np.concatenate(batch_pred_scores, axis=0)
            # # TODO [...共n1个0..., ...共n2个1..., ...共n3个2..., ......, ...共n20个19...,]
            tmp_pred_labels = np.concatenate(batch_pred_labels, axis=0)

            total_pred_boxes.append(tmp_pred_boxes)
            total_pred_scores.append(tmp_pred_scores)
            total_pred_labels.append(tmp_pred_labels)
            total_gt_boxes.append(batch_gt_boxes[0].cpu().numpy())
            total_gt_labels.append(batch_gt_labels[0].cpu().numpy())

        result = eval_detection_voc(
            total_pred_boxes,
            total_pred_labels,
            total_pred_scores,
            total_gt_boxes,
            total_gt_labels,
            iou_thresh=0.5,
            use_07_metric=True,
        )

        tqdm.write(f"Avg AP: {result['ap']}")
        tqdm.write(f"Avg mAP: {result['map']}")



if __name__ == "__main__":

    args = get_args_val()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = WSDDN(base_net=args.base_net)
    net.load_state_dict(torch.load(args.state_path))
    net.to(DEVICE)

    tqdm.write("State is loaded")

    test_ds = WSDDN_Dataset("test")  # len = 4952
    test_dl = DataLoader(test_ds, batch_size=None, shuffle=False, num_workers=8)
    evaluate(args, net, test_dl)
