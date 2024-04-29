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
        self.softmax = torch.nn.Softmax()
        self.resnet = models.resnet18(pretrained=True).cuda()

    def predict(self):
        inputI = '1.JPEG'

        img = Image.open(inputI)

        ans = self.transform(img, 224)
        ans = ans.cuda()
        if ans.data.size(1) == 3:
            ans = self.resnet(ans)
            print (ans)
            ans = self.softmax(ans)
            print(ans)
            ans = ans.data.cpu().numpy()[0]
            print (np.argmax(ans))

    def transform(self, img, size):
        # print (img.size)
        # img.show()
        ans = Scale(size)(img)
        # print (ans.size)
        # ans.show()
        ans = CenterCrop(size)(ans)
        # print (ans.size)
        # ans.show()
        return Variable(ToTensor()(ans)).view(1, -1, ans.size[1], ans.size[0])

d = deconver()
d.predict()