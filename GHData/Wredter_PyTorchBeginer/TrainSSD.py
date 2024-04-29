from __future__ import division

import os
from torch.utils.data import DataLoader
from Models.SSD.DefaultsBox import dboxes300
from Models.SSD.Utility import *
from Models.Utility.Utility import list_avg
from Models.Utility.DataSets import SSDDataset
from Models.SSD.SSD import *


if __name__ == "__main__":
    train = os.getcwd()
    train += "\\Data\\preped_data_mass_train.csv"
    test = os.getcwd()
    test += "\\Data\\preped_data_mass_test.csv"
    dummy_test = os.getcwd()
    dummy_test += "\\Data\\Dumy_test.csv"
    x = os.getcwd()
    x += "/Models/YOLO/config/yolov3.cfg"
    # test = ResourceProvider(y, "D:\\DataSet\\CBIS-DDSM\\", "D:\\DataSet\\ROI\\CBIS-DDSM\\")
    class_names = ["patologia"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoding = []
    num_classes = 1
    epochs = 200
    img_size = 300
    batch_size = 8
    loslist = []

    model = SSD300(num_classes).to(device)
    model = model.train(False)

    # na przyszłość nie robić tak jak zrobiłem to głupie i działa tylko dla konkretnego przypadku
    dumy_ds = SSDDataset(csv_file=test, img_size=img_size)
    dummy_loader = torch.utils.data.DataLoader(
        dumy_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=dumy_ds.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    start_epoch = 0
    iteration = 0
    t_bbox = dboxes300()
    loss_func = Loss(t_bbox, 0.5, num_classes).to(device)
    # Test input
    for batch_i, (imgs, targets) in enumerate(dummy_loader):
        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)
        # locations of targets
        targets_loc = targets[..., :4]
        # classes 0 for first
        targets_c = targets[..., -1]
        ploc, plabel = model(imgs)

        db = t_bbox(order="xywh").to(device)
        current_batch_size = ploc.shape[0]
        generate_plots(ploc, plabel, db, targets, imgs, current_batch_size, encoding, 0)# Only encoding is importatnt "raw","delta","d_delta"
    # Training
    optimizer.zero_grad()
    ds = SSDDataset(csv_file=train, img_size=img_size)
    model = model.train(True)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=ds.collate_fn,
    )
    for epoch in range(start_epoch, epochs):
        epoch_err = []
        for batch_i, (imgs, targets) in enumerate(dataloader):
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            # locations of targets
            targets_loc = targets[..., :4]
            # classes 0 for first
            targets_c = targets[..., -1]
            ploc, plabel = model(imgs)

            loss = loss_func(ploc, plabel, targets_loc, targets_c)
            if loss == 0:
                epoch_err.append(loss)
                continue
            else:
                epoch_err.append(loss.item())
                loss.backward()
            #if batch_i % 2:
            optimizer.step()
            optimizer.zero_grad()
        #scheduler.step()

        loslist.append(list_avg(epoch_err))
        if epoch % 5 == 0:
            print("Epoch: " + str(epoch) + " Total loss : " + str(loslist[-1])
                  )
    ptl.plot(loslist)
    ptl.ylabel("loss")

    ptl.show()
    z = os.getcwd()
    z += "\\Models\\SSD\\TrainedModel\\SSD_50_e100_t.pth"
    torch.save(model.state_dict(), z)
    """
    for batch_i, (imgs, targets) in enumerate(dummy_loader):
        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)
        # locations of targets
        targets_loc = targets[..., :4]
        # classes 0 for first
        targets_c = targets[..., -1]
        ploc, plabel = model(imgs)
        _, idx = plabel.max(1, keepdim=True)
        current_batch_size = ploc.shape[0]
        generate_plots(ploc, plabel, db, targets, imgs, current_batch_size, encoding, 0)
    """


    print("Skończyłem")

