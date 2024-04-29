from torchvision.models import resnext101_32x8d
import torch

# edit by Brilian 2020/03/05
body = resnext101_32x8d(pretrained=True)

# Use this one if you want to check the whole body of the model in easy way
# print('...base body: ', body)

# this is one way to cut unnecessary layer in your desired model as list, use with precaution
bodylist = list(body.children())#[:7] 
# this is a way to delete Sequential in the list of layer, use with precaution
bodylist[6] = bodylist[6][:1] 

# this is a way to list all of your remaining structure in a list
# for i, f in enumerate(bodylist):
#     print(i, f, '\n')
#     print('================================')

# if you want to check some specific layer and Sequential, use this
# print('...', bodylist[5])
# print('================================')
# print('...', bodylist[5][0])
# print('================================')

# Use this if you want to get the output channels of the respective layer
# I use it whenever I want to make FPN with C3, C4, C5
Clist = []
Clist += [body.layer2[-1].conv3.out_channels] # get C3
Clist += [body.layer3[-1].conv3.out_channels] # get C4
Clist += [body.layer4[-1].conv3.out_channels] # get C5
print('...Clist: ', Clist)

# ref: https://discuss.pytorch.org/t/accessing-intermediate-layers-of-a-pretrained-network-forward/12113/2
# Thanks for the reference, I use this to get my desired structures
# where we have:
#  - backbone: Resnext
#  - output: C3, C4, C5

class resnext101_custom(torch.nn.Module):
    def __init__(self, fs):
        super(resnext101_custom, self).__init__()
        features = fs
        bodylist = list(features.children())[:8]
        self.features = nn.ModuleList(bodylist)
        
    def forward(self, x, debug=False):
        results = []
        if debug: print('start...x.size(): ', x.size())
        for ii, fs in enumerate(self.features):
            x = fs(x)
            if ii >= 5:
                if debug: print('...x.size(): ', x.size())
                results +=[ x ]
        return results

# instantiate the class
body = resnext101_custom(body)
# check our new structure
print('...new body: ', body)

# for usage, you can use other repo for combination
# hope it helps ^,^
