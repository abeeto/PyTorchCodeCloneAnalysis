
class LaneDetector:
    def __init__(self, hnet, lanenet):
        self.hnet = hnet

        self.lanenet=lanenet

    def __call__(self, image, y_positions=None):
        """
        1 Apply segmentation network to the image
        2 Run DBSCAN over the embeddings for those pixels that are lanes.
        3 Apply h-net to the image
        4 Project pixel coordinates with the predicted homograpgy
        5 Fit the 3-rd order polynomial
        6 Predict the lane position for each provided $y$ (you should project this first).
        7 Compute back projection and return the positions of $x$ for each lane.
        :param image:
        :param y_positions:
        :return:
        """
        y1,y2=self.lanenet.forward(image)
        return y1,y2

if __name__=='__main__':
    from ENet import ENet
    from LaneDataset import LaneDataset
    import torch
    import cv2
    import numpy as np
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #model_path = 'lanenet_epoch_0_iter_0_batch_2.model'
    #model_path = 'lanenet_epoch_0_iter_150_batch_2.model'
    #model_path = 'lanenet_epoch_0_iter_300_batch_2.model'
    #model_path = 'lanenet_epoch_0_iter_740_batch_2.model'
    #model_path = 'lanenet_epoch_0_iter_210_batch_2.model'
    model_path = 'lanenet_epoch_1_iter_500_batch_2.model'


    lane_net = ENet()
    lane_net.load_state_dict(torch.load(model_path))

    ldet = LaneDetector(None, lane_net)


    #load dataset
    ldata  = LaneDataset('/data/tusimple', train=False)

    #get picture0
    data0 = ldata[0]
    data0=data0.reshape([1, data0.shape[0],data0.shape[1],data0.shape[2]])
    y1, y2=ldet(data0)


    img_org = ldata[0].detach().numpy()
    img_org = np.array(np.transpose(img_org, (1, 2, 0)), dtype=np.uint8)
    cv2.imshow('img', img_org)

    cv2.imshow('y11', y1.detach().numpy()[0,0])
    cv2.imshow('y12', y1.detach().numpy()[0, 1])

    y212345=y2.detach().numpy()[0]
    cv2.imshow('y21', y212345[0])
    cv2.imshow('y22', y212345[1])
    cv2.imshow('y23', y212345[2])
    cv2.imshow('y24', y212345[3])
    cv2.imshow('y25', y212345[4])

    cv2.waitKey(0)


