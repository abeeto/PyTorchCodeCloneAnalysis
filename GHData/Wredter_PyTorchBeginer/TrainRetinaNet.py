import os

import torch
import matplotlib.pyplot as ptl
from torch.autograd import Variable
from Models.RetinaNet.RetinaNet import RetinaNet
from Models.RetinaNet.Utility import nms_prep, retinabox300, retinabox600
from Models.SSD.Utility import compare_trgets_with_bbox
from Models.Utility.DataSets import SSDDataset
from Models.Utility.Utility import prep_paths, list_avg, point_form
from Models.RetinaNet.RLoss import RLoss


if __name__ == "__main__":
    train, test, dummy_test, class_names, yolo_cfg = prep_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoding = []
    loslist = []
    num_classes = 1
    epochs = 400
    img_size = 608
    batch_size = 2

    model = RetinaNet(num_classes).to(device)
    loss_func = RLoss(num_classes)
    model.freeze_bn()


    # na przyszłość nie robić tak jak zrobiłem to głupie i działa tylko dla konkretnego przypadku
    dumy_ds = SSDDataset(csv_file=dummy_test, img_size=img_size)
    dummy_loader = torch.utils.data.DataLoader(
        dumy_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=dumy_ds.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    start_epoch = 0
    iteration = 0
    t_bbox = retinabox600()
    db = t_bbox(order="xywh").to(device)
    # Test input
    for batch_i, (imgs, targets) in enumerate(dummy_loader):
        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)
        # locations of targets
        targets_loc = targets[..., :4]
        # classes 0 for first
        targets_c = targets[..., -1]
        ploc, plabel = model(imgs)

        nms_prep(imgs, targets, ploc, plabel, db)
    # Training
    optimizer.zero_grad()
    ds = SSDDataset(csv_file=train, img_size=img_size)

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=ds.collate_fn,
    )

    for epoch in range(start_epoch, epochs):
        print(f"--------------------- Epoch {epoch}/{epochs} ---------------------")
        epoch_err = []
        model = model.train(True)
        model.freeze_bn()
        for batch_i, (imgs, targets) in enumerate(dataloader):

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            # locations of targets
            targets_loc = targets[..., :4]
            # classes 0 for first
            targets_c = targets[..., -1]
            ploc, plabel = model(imgs)
            for batch_j in range(ploc.shape[0]):
                mached_loc, mached_label, mask = compare_trgets_with_bbox(t_bbox(order='ltrb').to(device),
                                                                          targets_loc[batch_j],
                                                                          targets_c[batch_j],
                                                                          0.4)
                if batch_j == 0:
                    m_pos = mached_loc.unsqueeze(0)
                    m_cls = mached_label.unsqueeze(0)
                    m_iou = mask.unsqueeze(0)
                else:
                    m_pos = torch.cat((m_pos, mached_loc.unsqueeze(0)), dim=0)
                    m_cls = torch.cat((m_cls, mached_label.unsqueeze(0)), dim=0)
                    m_iou = torch.cat((m_iou, mask.unsqueeze(0)), dim=0)


            m_pos = Variable(m_pos.to(ploc.device), requires_grad=False)
            m_cls = Variable(m_cls.to(ploc.device, dtype=torch.float32), requires_grad=False)
            m_iou = Variable(m_iou.to(ploc.device), requires_grad=False)
            pos_num = m_iou.long().sum().item()
            if pos_num == 0:
                print("Jeeeeez popraw boxy")
                break
            loss = loss_func(ploc, plabel, m_pos, m_cls, m_iou)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_i == 1:
                nms_prep(imgs, targets, ploc, plabel, db, epoch=epoch)
            epoch_err.append(loss.item())
            # if batch_i % 2:
        #scheduler.step()

        loslist.append(list_avg(epoch_err))
        if epoch % 5 == 0:
            print("Epoch: " + str(epoch) + " Total loss : " + str(loslist[-1])
                  )
    ptl.plot(loslist)
    ptl.ylabel("loss")

    ptl.show()
    """
    print("--------------------- TEST ---------------------")
    for batch_i, (imgs, targets) in enumerate(dummy_loader):
        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)
        # locations of targets
        targets_loc = targets[..., :4]
        # classes 0 for first
        targets_c = targets[..., -1]
        ploc, plabel = model(imgs)

        nms_prep(imgs, targets, ploc, plabel, db)
    """
    z = os.getcwd()
    z += "\\Models\\RetinaNet\\TrainedModel\\RetinaNet_400_lr10000.pth"
    torch.save(model.state_dict(), z)

    print("Skończyłem")

