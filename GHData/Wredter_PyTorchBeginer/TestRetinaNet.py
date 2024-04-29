import torch
import os
from torch.autograd import Variable

from Models.RetinaNet.RetinaNet import RetinaNet
from Models.RetinaNet.Utility import retinabox600, nms_prep
from Models.Utility.DataSets import SSDDataset
from Models.Utility.Utility import prep_paths
from Models.Utility.Metricks import Metrics

if __name__ == "__main__":
    train, test, dummy_test, class_names, yolo_cfg = prep_paths()
    trained_model = os.getcwd()
    trained_model += "\\Models\\RetinaNet\\TrainedModel\\RetinaNet_225e.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 1
    img_size = 608
    batch_size = 1
    t_bbox = retinabox600()
    quality = Metrics()
    db = t_bbox(order="xywh").to(device)
    with torch.no_grad():
        model = RetinaNet(num_classes)
        model.load_state_dict(torch.load(trained_model))
        model.to(device)
        model.eval()
        test_ds = SSDDataset(csv_file=test, img_size=img_size)
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
            targets_loc = targets[..., :4]
            targets_c = targets[..., -1]
            ploc, plabel = model(imgs)
            nms_prep(imgs, targets_loc, ploc, plabel, db)
            for batch_j in range(ploc.shape[0]):
                quality.statistic_prep(ploc[batch_j],
                                       plabel[batch_j],
                                       targets_loc[batch_j],
                                       targets_c[batch_j],
                                       db)
        print(quality.calc_net_stat())
