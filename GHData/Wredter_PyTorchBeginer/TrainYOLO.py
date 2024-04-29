from __future__ import print_function, division

from terminaltables import AsciiTable
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Models.Utility.DataSets import ImgDataset
from Models.Utility.ResourceProvider import *
from Models.Utility.Utility import list_avg
from Models.YOLO.darknet import Darknet
from Models.YOLO.utility import *
import matplotlib.pyplot as ptl

if __name__ == "__main__":
    train = os.getcwd()
    train += "\\Data\\preped_data_mass_train.csv"
    test = os.getcwd()
    test += "\\Data\\preped_data_mass_test.csv"
    x = os.getcwd()
    x += "/Models/YOLO/config/yolov3.cfg"
    dummy_test = os.getcwd()
    dummy_test += "\\Data\\Dumy_test.csv"
    # test = ResourceProvider(y, "D:\\DataSet\\CBIS-DDSM\\", "D:\\DataSet\\ROI\\CBIS-DDSM\\")
    class_names = ["patologia"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluation_interval = 10
    epochs = 150
    loslist = []
    model = Darknet(x).to(device)
    ds = ImgDataset(csv_file=train)
    img_size = 416

    dumy_ds = ImgDataset(csv_file=dummy_test, img_size=img_size)
    dummy_loader = torch.utils.data.DataLoader(
        dumy_ds,
        batch_size=4,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=dumy_ds.collate_fn,
    )
    for batch_i, (imgs, targets) in enumerate(dummy_loader):
        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)
        targets[:, 1] = targets[:, 1] - 1
        loss, outputs = model(imgs, targets)
        for batch_j in range(4):
            nms_prep(imgs[batch_j],targets[batch_j], outputs[batch_j])



    ############################## Train ########################################
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=ds.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(epochs):
        model.train()
        epoch_err = []
        for batch_i, (imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            targets[:, 1] = targets[:, 1] - 1

            loss, outputs = model(imgs, targets)
            loss.backward()
            epoch_err.append(loss.item())
            if epoch == 99:
                for batch_j in range(2):
                    nms_prep(imgs[batch_j], targets[batch_j], outputs[batch_j])

            if batches_done % 2:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j + 1}", metric)]
                tensorboard_log += [("loss", loss.item())]

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"
            print(log_str)

            model.seen += imgs.size(0)
        loslist.append(list_avg(epoch_err))

    ptl.plot(loslist)
    ptl.ylabel("loss")

    ptl.show()
    z = os.getcwd()
    z += "\\Models\\YOLO\\TrainedModel\\Yolov3_300e.pth"
    torch.save(model.state_dict(), z)

    print("Skończyłem")
