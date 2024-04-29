class transition_block(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(transition_block, self).__init__()
        
        self.relu=nn.ReLU(inplace=True)
        self.bn=nn.BatchNorm2d(num_features=out_channels)
        
        self.conv=nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size =1, bias=False)
        self.avg_pl=nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
    def forward(self, x):
        bn=self.bn(self.relu(self.conv(x)))
        output= self.avg_pl(bn)
        
        return output   
