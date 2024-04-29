

############################################

#width=512 height=256
import random
import time
import gc
from DiscriminativeLoss import DiscriminativeLoss
from ENet import ENet
from LaneDataset import LaneDataset

DEFAULT_SIZE = (256, 512)
import torch
from PIL import Image
import json
import os
import cv2
import numpy as np
import glob




##############################################



##################################################

##################################################
# TODO: Train segmentation and instance segmentation


from torch.autograd import Variable
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


def train_loop(dataloader, model, loss_fn, optimizer):
    global device
    learning_rate = 5e-4
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0002)

    is_plot_show_detail=False
    is_plot=False
    plot_iter=20
    plot_detail_iter=300
    if is_plot:
        f2, axarrorg = plt.subplots(4, 5)
    if is_plot_show_detail:
        f, axarr = plt.subplots(2, 4)

    # org
    # bin inst labels
    # bin dist
    # bin img
    # inst dist
    # inst img





    loss_record=[]
    for epoch in range(0,5):
        torch.cuda.empty_cache()

        model.train()
        t1=time.time()
        for iter, (X, seg_img, inst_img) in enumerate(dataloader):
            #if (iter %(int(random.randint(1,2))+1)==0):
            #    #skip some to speedup
            #    continue


            X = Variable(X)
            seg_img = Variable(seg_img)
            inst_img = Variable(inst_img)
            if torch.cuda.is_available():
                X=X.cuda()
                seg_img = seg_img.cuda()
                inst_img=inst_img.cuda()
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # Compute prediction and loss
                #print('model start', time.time()-t1)
                seg_out, inst_out = model(X)
                #print('loss start', time.time() - t1)
                if is_plot_show_detail and (iter % plot_detail_iter == 0):
                    bin_loss, inst_loss = loss_fn(seg_out, seg_img, inst_out, inst_img, plt_bin=axarr[0], plt_inst=axarr[1])
                else:
                    bin_loss, inst_loss = loss_fn(seg_out, seg_img, inst_out, inst_img)

            loss_all = bin_loss + inst_loss
            #print('Backpropagation start', time.time() - t1)

            # Backpropagation
            loss_all.backward()
            optimizer.step()

            if iter % 10 == 0:
                print('epoch:{}. iter:{}. bin_loss:{}. inst_loss:{}. loss_all:{}, time:{}'.format(epoch, iter, bin_loss, inst_loss,loss_all,time.time()-t1))
                loss_record.append(loss_all.item())
                torch.save(model.state_dict(),
                       f"lanenet_epoch_{epoch}_iter_{iter}_batch_{batch_size}.model")

                if is_plot and(iter%plot_iter==0):
                    axarrorg[0, 0].set_title(f'iter:{iter}')
                    img_org = np.array(np.transpose(np.squeeze(X.cpu().detach().numpy()[0]), (1, 2, 0)), dtype=np.uint8)
                    axarrorg[0, 0].imshow(img_org)
                    cv2.imshow('img_org', img_org)
                    img_gray = np.mean(img_org, axis=2)
                    axarrorg[1, 0].imshow(img_gray * 0.1 + seg_img.cpu().detach().numpy()[0] * 10)
                    axarrorg[1, 1].imshow(img_gray * 0.1 + inst_img.cpu().detach().numpy()[0]*100)

                    for i in range(len(seg_out[0])):
                        axarrorg[2, i].imshow(seg_out[0][i].cpu().detach().numpy())
                    for i in range(len(inst_out[0])) :
                        axarrorg[3, i].imshow(inst_out[0][i].cpu().detach().numpy())
                    #plt.draw()
                    #plt.show(block=False)
                    plt.pause(1)



        torch.save(model.state_dict(),
               f"lanenet_epoch_{epoch}_iter_{12345}_batch_{batch_size}.model")

    torch.save(model.state_dict(),
               f"lanenet_epoch_{12345}_iter_{12345}_batch_{batch_size}.model")

    to_rec=json.dumps(loss_record)
    f=open('lanenet_loss.txt', 'w')
    f.write(to_rec)
    f.close()


if __name__=='__main__':


    ldata = LaneDataset('/data/tusimple')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = 2
    print('curr device:', device)

    torch.cuda.empty_cache()

    model = ENet()
    #model_path = 'lanenet_epoch_1_iter_10_batch_2.model'
    #model.load_state_dict(torch.load(model_path))


    # 加载之前训练的数据
    model.to(device)

    loss_fn = DiscriminativeLoss()
    train_dataloader = DataLoader(ldata, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


    train_loop(train_dataloader, model, loss_fn, optimizer)
