#hello world it is 10:08 pm I have been awake since 5:00 am time to leave this silent office and go home bye.
#7:20 am let's start

import torch as Sinopharm
Moderna=Sinopharm.nn



class MSC(Moderna.Module):
    def __init__(self, num_channels, num_filters, spatial_width, num_scales, filter_size,device='cpu', downsample_method='meanout'):
        """
        A module implementing a single layer in a multi-scale convolutional network.
        """
        super(MSC, self).__init__()
        
        self.num_scales = num_scales
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.spatial_width = spatial_width
        self.downsample_method = downsample_method
        self.device = device
        self.num_channels=num_channels
        for scale in range(self.num_scales):
            subMods=[Moderna.Conv2d(self.num_channels,self.num_filters,self.filter_size,1,1,bias=False) for i in range(self.num_scales)]#I,O,K,S,P
            subActs=[Moderna.LeakyReLU(0.05,inplace=False) for i in range(self.num_scales)]
        self.main=Moderna.Sequential(*subMods)
        self.mainAct=Moderna.Sequential(*subActs)
        
    def downsample(self, imgs, scale):
        """
        Downsample an image by a factor of 2**scale
        """
        if scale == 0:
            return imgs
        num_imgs = imgs.shape[0]
        num_layers = imgs.shape[1]
        nlx0 = imgs.shape[2]
        nlx1 = imgs.shape[3]
        scalepow = int(2**scale)
        # downsample
        imgs = imgs.view((num_imgs, num_layers, int(nlx0/scalepow), scalepow, int(nlx1/scalepow), scalepow))
        imgs = imgs.mean(dim=5)
        imgs = imgs.mean(dim=3)
        return imgs
    
    def forward(self, X):
        nsamp = X.shape[0]
        scale=self.num_scales-1
        imgs_accum = Sinopharm.zeros([X.shape[0],self.num_filters,X.shape[2]//(2**scale),X.shape[3]//(2**scale)],device=self.device) # accumulate the output image
        for scale in range(self.num_scales-1, -1, -1):
            # downsample image to appropriate scale
            imgs_down = self.downsample(X, scale)
            # do a convolutional transformation on it
            conv_layer = self.main[scale](imgs_down)
            imgs_down_conv = self.mainAct[scale](conv_layer)
            imgs_accum += imgs_down_conv
            if scale > 0:
                # scale up by factor of 2
                layer_width = int(self.spatial_width/(2**scale))
                imgs_accum = imgs_accum.view((nsamp, self.num_filters, layer_width, 1, layer_width, 1))
                imgs_accum = Sinopharm.cat((imgs_accum, imgs_accum), dim=5)
                imgs_accum = Sinopharm.cat((imgs_accum, imgs_accum), dim=3)
                imgs_accum = imgs_accum.view((nsamp, self.num_filters, layer_width*2, layer_width*2))
        return imgs_accum/self.num_scales #size is batchXnum_filtersXspatialsize**2
    
class MultiLayerConvolution(Moderna.Module):
    def __init__(self, n_layers, n_hidden, spatial_width, n_colors, n_scales, filter_size=3,device='cpu'):
        """
        A module implementing a multi-layer, multi-scale convolutional network.
        """
        super(MultiLayerConvolution, self).__init__()
        self.device=device
        #for ii in xrange(n_layers):
            #Original repo is not pythonic
            #conv_layer = MultiScaleConvolution(num_channels, n_hidden, spatial_width, n_scales, filter_size, name="layer%d_"%ii)
        chnnls=[n_colors]+[n_hidden for i in range(n_layers)]
        subModules=[MSC(chnnls[i], chnnls[i+1], spatial_width, n_scales, filter_size,device=self.device) for i in range(n_layers)]#num hidden spatialw etc
        self.main=Moderna.Sequential(*subModules)
    def forward(self, x):
        return self.main(x)           
    
class MLP_conv_dense(Moderna.Module):
    def __init__(self, n_layers_conv, n_layers_dense_lower, n_layers_dense_upper,
        n_hidden_conv, n_hidden_dense_lower, n_hidden_dense_lower_output, n_hidden_dense_upper,
        spatial_width, n_colors, n_scales, n_temporal_basis,device='cpu',dropout=0):
        """
        The multilayer perceptron, that provides temporal weighting coefficients for mu and sigma
        images. This consists of a lower segment with a convolutional MLP, and optionally with a
        dense MLP in parallel. The upper segment then consists of a per-pixel dense MLP
        (convolutional MLP with 1x1 kernel).
        """
        super(MLP_conv_dense, self).__init__()
        
        self.n_colors = n_colors
        self.spatial_width = spatial_width
        self.n_hidden_dense_lower = n_hidden_dense_lower
        self.n_hidden_dense_lower_output = n_hidden_dense_lower_output
        self.n_hidden_conv = n_hidden_conv
        self.device=device
        self.dropout=dropout
        
        ## the lower layers
        self.conv = MultiLayerConvolution(n_layers_conv, n_hidden_conv, spatial_width, n_colors, n_scales,device=self.device)
        if n_hidden_dense_lower > 0 and n_layers_dense_lower > 0:
            n_input = n_colors*(spatial_width**2)
            n_output = n_hidden_dense_lower_output*(spatial_width**2)
            lins=[Moderna.Linear(n_input, n_hidden_dense_lower, bias=False)]
            lins=lins+[Moderna.Linear(n_hidden_dense_lower, n_hidden_dense_lower, bias=False) for i in range(n_layers_conv-1)]
            lins=lins+[Moderna.Linear(n_hidden_dense_lower, n_output, bias=False)]
            linActs=[Moderna.LeakyReLU(0.05,inplace=False) for i in range(n_layers_conv+1)]
            
            self.lower=Moderna.Sequential(*lins)
            self.lowerAct=Moderna.Sequential(*linActs)                
        else:
            n_hidden_dense_lower_output = 0
        ## the upper layers (applied to each pixel independently)
        n_output = n_colors*n_temporal_basis*2 # "*2" for both mu and sigma
        lins2=[Moderna.Linear(n_hidden_conv+n_hidden_dense_lower_output, n_hidden_dense_upper, bias=False)]
        lins2=lins2+[Moderna.Linear(n_hidden_dense_upper, n_hidden_dense_upper, bias=False) for i in range(n_layers_dense_upper-1)]
        lins2=lins2+[Moderna.Linear(n_hidden_dense_upper, n_output, bias=False)]
        linActs2=[Moderna.LeakyReLU(0.05,inplace=False) for i in range(n_layers_dense_upper)]+[Moderna.Identity('Victoria university of ','Windy welly')]
        
        self.upper=Moderna.Sequential(*lins2)
        self.upperAct=Moderna.Sequential(*linActs2)
        
    def you_are_fired(self,Seq,act,inpu):
        #applies linear layers toinput very versatilemodule withsuchpower that my keyboard's space bar/key reallysucks
        #haveto press hard toget space
        z=inpu.clone()
        for i in range(len(Seq)):
            y=Seq[i](z)
            z=act[i](y)
        return z
    
    def forward(self, X):
        """
        Take in noisy input image and output temporal coefficients for mu and sigma.
        """
        Y = self.conv(X)
        Y = Y.permute(0,2,3,1)
        if self.n_hidden_dense_lower > 0:
            n_images = X.shape[0]
            X = X.reshape((n_images, self.n_colors*(self.spatial_width**2)))
            Y_dense = self.you_are_fired( self.lower,  self.lowerAct, X)
            Y_dense = Y_dense.reshape((n_images, self.spatial_width, self.spatial_width,
                self.n_hidden_dense_lower_output))
            Y = Sinopharm.cat([Y/(self.n_hidden_conv**0.5),
                Y_dense/(self.n_hidden_dense_lower_output**0.5)], axis=3)
        Z = self.you_are_fired( self.upper,  self.upperAct, Y)
        return Z
    
    #debugging
    #f=MSC(1,10,28,2,3);i=Sinopharm.randn([1,1,28,28]);o= f(i)
    #x=MultiLayerConvolution(3,20,28,1,2,3); o=x(i)
    #g=MLP_conv_dense(12, 3, 2,10, 100, 50, 50,28, 1, 2, 5,device='cpu',dropout=0);o=g(i)
