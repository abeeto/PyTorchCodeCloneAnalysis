import os

import torch
from torch.autograd import Variable

from Models.SSD.DefaultsBox import dboxes300
from Models.SSD.SSD import SSD300
from Models.SSD.Utility import generate_plots
from Models.Utility.DataSets import SSDDataset
from Models.Utility.Utility import prep_paths
from Models.Utility.Metricks import Metrics

if __name__ == "__main__":
    train, test, dummy_test, class_names, yolo_cfg = prep_paths()
    trained_model = os.getcwd()
    trained_model += "\\Models\\SSD\\TrainedModel\\SSD_50_e200_t.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoding = []
    num_classes = 1
    img_size = 300
    batch_size = 8
    t_bbox = dboxes300()
    db = t_bbox(order="xywh").to(device)
    quality = Metrics()
    with torch.no_grad():
        model = SSD300(num_classes)
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
            cpy_ploc = ploc.clone()
            cpy_plabel = plabel.clone()
            current_batch_size = ploc.shape[0]
            generate_plots(ploc, plabel, db, targets, imgs, current_batch_size, encoding, 0)
            for batch_j in range(ploc.shape[0]):
                quality.statistic_prep(cpy_ploc[batch_j],
                                       cpy_plabel[batch_j],
                                       targets_loc[batch_j],
                                       targets_c[batch_j],
                                       db)
        quality.calc_net_stat()
