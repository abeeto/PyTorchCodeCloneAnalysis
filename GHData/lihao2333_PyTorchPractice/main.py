import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


from net import Net
from data_base import DataBase
from trainer import Trainer
from tester import Tester

class ArgParser(object):

    def __init__(self):
        self._parser = argparse.ArgumentParser(description="PyTorch Tutorial")
        self._parser.add_argument("--train", action="store_true")
        self._parser.add_argument("--test", action="store_true")
        self._args = self._parser.parse_args()

    def is_train(self):
        return self._args.train

    def is_test(self):
        return self._args.test


if __name__ == "__main__":
    arg_parser = ArgParser()
    data_base = DataBase("./data")
    NET_PATH="./cifar_net.pth"
    if arg_parser.is_train():
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        trainer = Trainer().\
            set_optimizer(optimizer).\
            set_num_epochs(2).\
            set_train_loader(data_base.train_loader()).\
            set_net(net).\
            set_criterion(criterion)

        trainer.train()
        torch.save(net.state_dict(), NET_PATH)


    if arg_parser.is_test():
        net = Net()
        net.load_state_dict(torch.load(NET_PATH))
        tester = Tester().\
                set_test_loader(data_base.test_loader()).\
                set_net(net)
        tester.test()


