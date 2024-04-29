from __future__ import print_function
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import CenterCrop, ToTensor, Scale

import torch.tensor
import numpy as np
import torchvision.models as models

class deconver:
    def __init__(self):
        self.alexnet = models.alexnet(pretrained=True).cuda()
        self.softmax = torch.nn.Softmax()

        self.ground_truth = []
        f = open('ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt', 'r')
        for line in f.readlines():
            self.ground_truth.append(int(line.strip()))
        f.close()
        # self.resnet18 = models.resnet18(pretrained=True)
        # self.vgg11 = models.vgg11(pretrained=True)
        # self.squeezenet1_0 = models.squeezenet1_0(pretrained=True)
        # self.densenet121 = models.densenet121(pretrained=True)
        # self.inception = models.inception_v3(pretrained=True)

    def test(self):
        self.counter = 0
        self.input = 'ILSVRC2012_img_val/ILSVRC2012_val_'
        for i in xrange(6666):
        # for i in xrange(len(self.ground_truth)):
            self.precit(i)

    def getInput(self, index):
        self.input
        s = "%08d" % (index+1)
        return self.input+s+'.JPEG'

    def precit(self, index):
        inputI = self.getInput(index)

        img = Image.open(inputI)

        ans = self.transform(img, 224)
        ans = ans.cuda()
        if ans.data.size(1) == 3:
            ans = self.alexnet(ans)
            ans = self.softmax(ans)
            # print(ans)
            ans = np.argmax(ans.data.cpu().numpy())

            ans = ans - self.ground_truth[index]
            if ans == 0:
                self.counter += 1
            print(ans)

    def transform(self, img, size):
        print (img.size)
        # img.show()
        ans = Scale(size)(img)
        # print (ans.size)
        # ans.show()
        ans = CenterCrop(size)(ans)
        # print (ans.size)
        # ans.show()
        return Variable(ToTensor()(ans)).view(1, -1, ans.size[1], ans.size[0])


d = deconver()
d.test()
print ('Total accuracy: {}, counter: {}'.format(d.counter/len(d.ground_truth), d.counter))
