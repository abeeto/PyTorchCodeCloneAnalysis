from DiscriminativeLoss import DiscriminativeLoss
from LaneDataset import LaneDataset
import torch
import cv2
import numpy as np
from ENet import ENet
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS
from LaneNetCluster import LaneNetCluster

if __name__ == '__main__':


    f, axarr = plt.subplots(2, 4)

    #######################
    #load model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model_path = 'lanenet_epoch_0_iter_0_batch_2.model'
    # model_path = 'lanenet_epoch_0_iter_150_batch_2.model'
    # model_path = 'lanenet_epoch_0_iter_300_batch_2.model'
    # model_path = 'lanenet_epoch_0_iter_740_batch_2.model'
    # model_path = 'lanenet_epoch_0_iter_210_batch_2.model'
    model_path = 'lanenet_epoch_2_iter_550_batch_2.model'

    lane_net = ENet()
    lane_net.load_state_dict(torch.load(model_path))
    lane_net.to(device)

    # load dataset

    ldata = LaneDataset('/data/tusimple', train=True)
    # get picture0
    data0, bin_img, inst_img = ldata[11]

    #####################################
    #prepare data
    data0 = data0.reshape([1, data0.shape[0], data0.shape[1], data0.shape[2]])
    bin_img = torch.unsqueeze(bin_img, 0)
    inst_img = torch.unsqueeze(inst_img, 0)
    if torch.cuda.is_available():
        data0 = data0.cuda()
        bin_img = bin_img.cuda()
        inst_img = inst_img.cuda()


    #######################################
    # calc data
    bin_logits, inst_logits = lane_net.forward(data0)
    # y1[y1>1] = 1
    # y1[y1<0] = 0

    loss_fn = DiscriminativeLoss()
    seg_loss, inst_loss = loss_fn(bin_logits,  bin_img, inst_logits, inst_img, plt_bin=axarr[0], plt_inst=axarr[1],plt_skip_zero_point=True)

    ###########################################
    # image show

    bin_img =  torch.squeeze(bin_img).cpu().detach().numpy()
    inst_img=  torch.squeeze(inst_img).cpu().detach().numpy()
    img_org = data0.cpu().detach().numpy()
    img_org = np.array(np.transpose(np.squeeze(img_org), (1, 2, 0)), dtype=np.uint8)
    # cv2.imshow('img', img_org)
    img_gray = np.mean(img_org, axis=2)
    bin_logits = bin_logits.cpu().detach().numpy()[0]

    y212345 = inst_logits.cpu().detach().numpy()[0]
    inst_logits = np.linalg.norm(x=y212345, axis=0)

    yy = np.transpose(y212345, (1, 2, 0))
    # yy=cv2.resize(yy,(yy.shape[1]//4,yy.shape[0]//4), interpolation=cv2.INTER_NEAREST)
    yy2 = yy.reshape(yy.shape[0] * yy.shape[1], yy.shape[2])


    f.savefig('./pics/instance_logit_distance.png')
    f, axarr = plt.subplots(3, 5)
    axarr[0, 0].imshow(img_org)
    axarr[0, 1].imshow(img_gray * 0.1 + bin_img * 100)
    axarr[0, 2].imshow(img_gray * 0.1 + inst_img)


    axarr[1, 0].imshow(bin_logits[0] * 100)
    axarr[1, 1].imshow(bin_logits[1] * 100)
    axarr[1, 2].imshow(bin_logits[0]>3.5)


    axarr[2, 0].imshow(y212345[0])
    axarr[2, 1].imshow(y212345[1])
    axarr[2, 2].imshow(y212345[2])
    axarr[2, 3].imshow(y212345[3])
    axarr[2, 4].imshow(y212345[4])
    #axarr[2, 0].imshow(img_gray * 0.1 + inst_img)
    # axarr[3].imshow(yy3)

    lnc = LaneNetCluster()
    mask_image, lane_lines, cluster_index = lnc.get_lane_lines(bin_logits, y212345)
    #lnc.get_lane_mask()
    axarr[0, 3].imshow(mask_image)
    f.savefig('./pics/draw_test.png')


    plt.show()

    cv2.waitKey(0)
