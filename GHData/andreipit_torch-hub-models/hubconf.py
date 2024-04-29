dependencies = ['torch','os']
from torchvision.models.resnet import resnet18 as _resnet18

def _hidden_in_hub_list(**kwargs):
    for key, value in kwargs.items():
        print("{0} = {1}".format(key, value))
    return


def shown_in_hub_list(**kwargs):
    """model = torch.hub.load('andreipit/torch-hub-models:master', 'shown_in_hub_list', pretrained=True, testkwarg=78)"""
    for key, value in kwargs.items():
        print("{0} = {1}".format(key, value))
    print('v1')        
    return kwargs.items()

# resnet18 is the name of entrypoint
def one_layer_cnn(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    Resnet18 model
    pretrained (bool): kwargs, load pretrained weights into the model

    to save weights in kaggle use: 
        PATH = './one_layer_cnn_weights.pth'
        torch.save(net.state_dict(), PATH)
    """
    # Call the model, load pretrained weights

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import os

    class Net(nn.Module):
        def __init__(self, _input_size=(3,28,28)):
            super(Net, self).__init__()
            #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            _filter_size = 5
            _padding = (int)((_filter_size - 1) / 2) # add zeros at borders -> to keep resolution
            _stride = 1 # do nothing (gap between filter applying)
            _in = 3
            _out = 6
            self.conv1 = nn.Conv2d(in_channels=_in, out_channels=_out, kernel_size=(_filter_size,_filter_size),padding=(_padding,_padding), stride=(_stride,_stride)) 
            _pool_size=2
            self.pool = nn.MaxPool2d(_pool_size, _pool_size)
    #         self.conv2 = nn.Conv2d(6, 16, 5)
            #self.fc1 = nn.Linear(16 * 5 * 5, 120)
            #self.pixels_after_conv = (int)((28/_pool_size)*(28/_pool_size)*_out)
            self.pixels_after_conv = (int)((_input_size[1]/_pool_size)*(_input_size[2]/_pool_size)*_out)
            self.fc1 = nn.Linear(self.pixels_after_conv, 120)
    #         self.fc1 = nn.Linear(-1, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
        def forward(self, x):
            debug = False  
            if debug: print('1',x.shape) # [4, 3, 28, 28]) 4->BatchSize
            x = self.pool(F.relu(self.conv1(x))) 
            if debug: print('2',x.shape) # [4, 6, 14, 14])
            #x = x.view(-1,self.pixels_after_conv) 
            x = x.view(x.shape[0],-1)  # x.shape[0] = 4 (batch size), -1 => WxHxC
            if debug: print('3',x.shape) # [4, 1176] # reshape
            x = F.relu(self.fc1(x)) 
            if debug: print('4',x.shape) # [4, 120]
            x = F.relu(self.fc2(x)) 
            if debug: print('5',x.shape) # [4, 84]
            x = self.fc3(x) 
            if debug: print('6',x.shape) # [4, 10] 4->BatchSize !!!
            return x

    # model = _resnet18(pretrained=pretrained, **kwargs)
    # net = Net(_input_size=iter(trainloader).next()['image'].shape[1:]) #(4,3,28,28)->(3,28,28)
    input_size = None
    for key, value in kwargs.items():
        if key=='input_size': input_size=value
    
    net = Net(_input_size=input_size) #(4,3,28,28)->(3,28,28)
    model = net




    if pretrained:
        # For checkpoint saved in local github repo, e.g. <RELATIVE_PATH_TO_CHECKPOINT>=weights/save.pth
        dirname = os.path.dirname(__file__)
        checkpoint = os.path.join(dirname, 'one_layer_cnn_weights.pth') # <RELATIVE_PATH_TO_CHECKPOINT>
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)


        # For checkpoint saved elsewhere
        #checkpoint = 'https://github.com/andreipit/torch-hub-models/releases/download/v1/one_layer_cnn_weights.pth'
        #model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False))

        # checkpoint = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
        # model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False))


    return model