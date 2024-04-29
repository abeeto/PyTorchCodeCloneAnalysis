from __future__ import division

import os

from torch.autograd import Variable
from torch.utils.data import DataLoader

from Models.Utility.DataSets import *
from Models.Utility.Utility import prep_paths
from Models.YOLO.darknet import *
from Models.Utility.Metricks import Metrics


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ImgDataset(path, img_size=img_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    train, test, dummy_test, class_names, yolo_cfg = prep_paths()
    trained_model = os.getcwd()
    trained_model += "\\Models\\YOLO\\TrainedModel\\Yolov3_300e.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 1
    batch_size = 8
    quality = Metrics()
    with torch.no_grad():
        model = Darknet(yolo_cfg)
        model.load_state_dict(torch.load(trained_model))
        model.to(device)
        model.eval()
        test_ds = ImgDataset(csv_file=train)
        dataloader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            collate_fn=test_ds.collate_fn,
        )

        for batch_i, (imgs, targets) in enumerate(dataloader):
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            targets[:, 1] = targets[:, 1] - 1

            loss, outputs = model(imgs, targets)
            for batch_j in range(outputs.shape[0]):
                quality.stastic_prep_YOLO(outputs[batch_j, :, :4], outputs[batch_j, :, 4], targets[batch_j], imgs[batch_j])
                # output_loc, output_cls, target_loc, num = nms_prep(imgs[batch_j], targets[batch_j], outputs[batch_j])
            print(f'Batch: {batch_i}')
    quality.calc_net_stat()
    print("finish")

