from DiscriminativeLoss import DiscriminativeLoss
from LaneDataset import LaneDataset
import torch
import cv2
import numpy as np
from ENet import ENet
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS
from LaneNetCluster import LaneNetCluster
import sys

DEFAULT_SIZE = (256, 512)


def run_image_once(lane_net, data0):

    # load dataset

    #######################################
    # calc data
    bin_logits, inst_logits = lane_net.forward(data0)

    ###########################################
    # image show

    img_org = data0.cpu().detach().numpy()
    img_org = np.array(np.transpose(np.squeeze(img_org), (1, 2, 0)), dtype=np.uint8)
    # cv2.imshow('img', img_org)
    img_gray = np.mean(img_org, axis=2)
    bin_logits = bin_logits.cpu().detach().numpy()[0]
    y212345 = inst_logits.cpu().detach().numpy()[0]

    lnc = LaneNetCluster()
    mask_image, lane_lines, cluster_index = lnc.get_lane_lines(bin_logits, y212345)

    return mask_image, lane_lines, cluster_index


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()

    if len(sys.argv)<2:
        model_path = 'lanenet_epoch_2_iter_550_batch_2.model'
    else:
        model_path = sys.argv[1]

    if len(sys.argv)<3:
        #data0 = cv2.imread('/data/tusimple/clips/0531/1492626270684175793/1.jpg')
        data0 = cv2.imread('./pics/1.jpg')
        data0 = cv2.resize(data0, (DEFAULT_SIZE[1], DEFAULT_SIZE[0]))
        data0 = np.array(np.transpose(data0, (2, 0, 1)), dtype=np.float32)
        data0 = data0.reshape([1, data0.shape[0], data0.shape[1], data0.shape[2]])
    else:
        data0 = cv2.imread(sys.argv[2])
        data0 = cv2.resize(data0, (DEFAULT_SIZE[1], DEFAULT_SIZE[0]))
        data0 = np.array(np.transpose(data0, (2, 0, 1)), dtype=np.float32)
        data0 = data0.reshape([1, data0.shape[0], data0.shape[1], data0.shape[2]])

    data0 = torch.from_numpy(data0).cuda()

    if len(sys.argv)<4:
        is_plot=1
    else:
        is_plot = int(sys.argv[3])

    #######################
    #load model

    lane_net = ENet()
    lane_net.load_state_dict(torch.load(model_path))
    lane_net.to(device)

    mask_image, lane_lines, cluster_index = run_image_once(lane_net, data0)

    if is_plot:
        f, axarr = plt.subplots(len(lane_lines)+2, 1)
        axarr[0].imshow(cv2.imread('./pics/1.jpg')) #FIXME: this should be replace with real data???
        axarr[1].imshow(mask_image)

        for i in range(len(lane_lines)):
            axarr[2+i].plot(lane_lines[i][1],lane_lines[i][0])
            axarr[2+i].set_xlim([0, DEFAULT_SIZE[1]])
            axarr[2 + i].set_ylim([DEFAULT_SIZE[0],0])
            axarr[2 + i].set_aspect('equal')
        plt.savefig('./pics/run_resultimage.png')
        plt.show()

