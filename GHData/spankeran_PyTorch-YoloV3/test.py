#-*-coding:utf-8-*-
import os.path as osp
import cv2
import torch
from torch.utils.data import DataLoader
import numpy as np
import prettytable as pt
import argparse
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = osp.dirname(osp.abspath(__file__))

from models.yolo_network import yolo_network
from data.dataset import VOCDataset, VOC_CLASSES, COCO_CLASSES
import utils.bbox
import utils.logger
import utils.metric
import utils.utils

logger = utils.logger.logger()

dataset_list = {
    'VOC': VOCDataset,
    'COCO': None
}

def test() :
    parser = argparse.ArgumentParser(
        prog='Pytorch-Ssd',
        description='a simple implementation of SSD(Single Shot MultiBox Detector) in PyTorch'
    )
    parser.add_argument('-dt', '--dataset_type', type=str, default='VOC', choices=('VOC', 'COCO'), help='dataset type')
    parser.add_argument('-td', '--test_dataset', type=tuple, default=((2012, 'test'),), help='test dataset')
    parser.add_argument('-is', '--input_size', type=tuple, default=(416, 416), help='ths image size of network input')
    parser.add_argument('-mn', '--model_name', type=str, default="yolov3-spp", help="the name of model")
    parser.add_argument('-wn', '--weight_name', type=str, help='the name weight file')
    parser.add_argument('-bs', '--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('-mt', '--conf_threshold', type=float, default=0.1,
                        help='threshold of the confidence when predicting')
    parser.add_argument('-mt', '--nms_threshold', type=float, default=0.4,
                        help='threshold of the iou when predicting')
    parser.add_argument('-rd', '--result_dirname', type=str, default=osp.join(BASE_DIR, 'results'),
                        help='the directory of results')
    parser.add_argument('-wed', '--weight_dirname', type=str, default=osp.join(BASE_DIR, 'weights'),
                        help='the directory of weights')
    args = parser.parse_args()

    utils.utils.checkdir(args.result_dir)
    utils.utils.checkdir(args.weight_dir)
    utils.utils.checkdir(osp.join(args.result_dir, 'predict'))

    dataset = dataset_list[args.dataset_type](
        dataset_name=args.train_dataset,
        input_size=args.input_size,
        multi_scale=False
    )

    dataloader = DataLoader(
        dataset=dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_works, pin_memory=True,
        collate_fn=dataset.collate_fn()
    )

    ssd_net = yolo_network(
        network_name=args.model_name, class_num=dataset.get_class_num(),
        ignore_thresold=None, conf_thresold=args.conf_thresold, nms_thresold=args.nms_thresold,
        coord_scale=None, conf_scale=None, cls_scale=None, device=device
    ).to(device)
    ssd_net.load_state_dict(torch.load(osp.join(args.weight_dirname, args.weight_name), map_location=device))

    results = dict()
    for class_name in dataset.class_names :
        results[class_name] = list()

    logger.info("Start testing...")
    pbar = tqdm(total=len(dataloader), desc="Test:")
    for batch_idx, (imgs, _, img_ids) in enumerate(dataloader) :
        imgs = imgs.to(device)

        with torch.no_grad() :
            predcitions = ssd_net(imgs)
            for prediction in predcitions :
                if prediction :
                    for p in prediction :
                        results[dataset.class_names[p[-1].long().item()]].append(
                            p.cpu().numpy().tolist()
                        )
        pbar.update(1)

    pbar.close()

    logger.info("Saving results")
    for class_name, result in tqdm(results.items(), desc="Saving:") :
        if len(result) :
            result = np.array(result)
            np.savetxt(
                fname=osp.join(args.result_dir, 'predict', '%s.txt' % class_name),
                X=result, fmt="%.4f",
                header="min_x  min_y  max_x  max_y scores class_id"
            )

    logger.info("Test done")

def evaluate(model, type, setname, part_num, batch_size, device):
    model.eval()

    dataset = dataset_list[type](
        setname=setname,
        transform=None,
        multi_scale=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=dataset.collate_fn(), pin_memory=True
    )

    labels = []
    sample_metrics = []
    for batch_i, (imgs, targets, _) in enumerate(dataloader) :
        imgs = imgs.to(device)
        targets = targets.to(device)

        if batch_i >= part_num :
            break

        labels += targets[:, 1].cpu().numpy().tolist()
        targets[:, 2:] = utils.bbox.xywh2xyxy(targets[:,2:])

        with torch.no_grad():
            outputs = model(imgs)

        sample_metrics += utils.metric.get_batch_statistics(outputs, targets, iou_threshold=.5)

    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, unique_class = utils.metric.ap_per_class(true_positives, pred_scores, pred_labels, labels)

    tb = pt.PrettyTable(field_names=["class", "precision", "recall", "f1", "AP"])
    for class_id in range(unique_class.shape[0]) :
        tb.add_row([dataset.class_names[class_id],
                    round(precision[class_id], 2),
                    round(recall[class_id],2),
                    round(f1[class_id], 2), round(AP[class_id], 2)])
    tb.add_row(['mean',
                round(precision.mean(), 2),
                round(recall.mean(), 2),
                round(f1.mean(), 2), round(AP.mean(), 2)])

    return '\n' + tb.get_string()

def predict_ong_image(model_cfg_path, img_path, weights_path, device, data='VOC') :
    class_num = 20 if data == 'VOC' else 80
    id2classname = VOC_CLASSES if data == 'VOC' else COCO_CLASSES

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = img.copy()
    img = cv2.resize(img, (416, 416))
    img = img.transpose(2, 0, 1) / 255.
    img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)

    net = yolo_network(model_cfg_path, class_num,
                ignore_thresold=.5, conf_thresold=.1, nms_thresold=.3,
                 coord_scale=5., conf_scale=.5, cls_scale=1.0, device=device)
    if '.weights' in weights_path :
        net.load_weights(weights_path)
    elif 'state' in weights_path :
        state_dicts = torch.load(weights_path,
                                map_location=device)
        net.load_state_dict(state_dicts)
    else :
        state_dicts = torch.load(weights_path,
                            map_location=device)
        net.load_state_dict(state_dicts['state_dicts'][0])
    net.eval()

    with torch.no_grad() :
        pred = net(img)

    for image_idx, bbox in enumerate(pred) :
        if bbox is not None :
            class_ids = bbox[:, -1].view(-1).numpy()
            scores = bbox[:, -2].view(-1).numpy()
            boxes = bbox[:, :4].numpy()
            classnames = [id2classname[int(i)] for i in class_ids]

            utils.bbox.plt_bboxes(input_img, classnames, scores, boxes)

if __name__ == '__main__' :
    model_cfg_path = r'./models/cfg/yolov3-spp.cfg'
    img_path = r'../ssd/test/person2.jpg'
    weight_path = r'./weights/yolov3-spp_state_2020-04-22-14-26-47.pth'

    predict_ong_image(model_cfg_path, img_path, weight_path, torch.device('cpu'), data='VOC')