from MyNet import MyNet
from nets.GoogLeNet import Inception
import torch
import time

if __name__ == '__main__':
    net = MyNet(net_type="VGGNet", dimensions=3, epoch=3, batch_size=8, img_size=96)  # 85 seconds

    # net = MyNet(net_type="DenseNet")  # 45 seconds
    # net = /MyNet(net_type="ResNet") # 50 seconds
    # net = MyNet(net_type="GoogLeNet") # 63 seconds
    # net = MyNet(net_type="GoogLeNet", img_size=224) # 85 seconds

    start = time.time()
    net.train()
    end = time.time()
    print(" - Training lasted\033[1m {:4.2f} seconds\033[0m".format(end - start))

    # net.test()
    net.canny_test(threshold=5)
