import  torchvision
import torch

from  label_shower import LabelShower


class Tester(object):

    def __init__(self):
        self._test_loader = None
        self._net = None

    def set_test_loader(self, test_loader):
        self._test_loader = test_loader
        return self

    def set_net(self, net):
        self._net = net
        return self

    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self._test_loader:
                images, labels = data
                outputs = self._net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("accuracy of the network on the 10000 test images: %d %%"%(
            100 * correct / total
            ))
        # data_iter = iter(self._test_loader)
        # images, labels = data_iter.next()
        # # import cv2
        # # cv2.imshow(torchvision.utils.make_grid(images))
        # label_shower = LabelShower()
        # label_shower.show("ground_truth",labels)

        # outputs = self._net(images)
        # _, predicted = torch.max(outputs, 1)
        # label_shower.show("predicted", predicted)


