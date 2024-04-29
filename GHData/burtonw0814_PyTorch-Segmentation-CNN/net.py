import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from layers2 import *














class Net(nn.Module):

    def __init__(self, input_depth, num_classes, device='cpu'):
        super(Net, self).__init__()
        
        self.params=[];
        self.f_size=3;
        self.input_depth=input_depth;
        self.num_classes=num_classes;
        self.device=device;
        self.norm=True
        self.embed_dim=1024;
        self.const=torch.tensor(0.001).to(self.device)   
        self.stvs=0.01 
        self.dim1=torch.tensor(1).to(self.device)
        self.dim2=torch.tensor(2).to(self.device) 

        self.pre_conv=self.conv2d_def(self.input_depth, 
                                            3, 
                                            self.f_size)

        ###########################
        ## Resnet feature extractor
        ###########################
        self.res_new, self.param_list, self.drop_layers, self.out_dim_list=get_resnet18_extractor(self.params)
        self.num_drops=len(self.drop_layers)        
        ###########################
        ###########################

        print('drop layers and out dim')
        print(self.drop_layers, self.out_dim_list)

        ##################
        # Bottleneck layer
        ##################
        self.bottle_neck=self.ASPP_2d_def(self.out_dim_list[-1], 
                                            self.out_dim_list[-1], 
                                            self.f_size)
        self.bottle_neck_atn=self.self_atn_2d_def(self.out_dim_list[-1], self.out_dim_list[-1], 128)
        self.bottle_neck2=self.ASPP_2d_def(self.out_dim_list[-1], 
                                            self.out_dim_list[-1], 
                                            self.f_size)
        
        '''self.bottle_neck3=self.ASPP_2d_def(self.out_dim_list[-1], 
                                            self.out_dim_list[-1], 
                                            self.f_size)'''
        ##################
        ##################

        # Option to add memory modules here

        ###################
        ##### DECODER #####
        ###################
        # Generate decoder parameters for pre-specified decoder that we care about
        self.decoder_param_list=[]; self.up_list=[];
        self.embed_levels=[]; # Construct number of channels based on depth of feature extractor
        self.atn_list=[];
        for i in range(len(self.drop_layers)):
            #self.embed_levels.append(int(self.v_embed_dim/(2**i)))
            self.embed_levels.append(self.out_dim_list[-1-i])
        self.embed_levels.append(input_depth)    

        for j in range(len(self.drop_layers)): 
            temptemp=[]; # Each convs in that level


            if False: #j>1 and j<(len(self.drop_layers)-2):
                conv_func=self.ASPP_2d_def 
            else:
                conv_func=self.conv2d_def 

               
            temptemp.append(conv_func(self.embed_levels[j]*2, 
                                    self.embed_levels[j], self.f_size))
            temptemp.append(conv_func(self.embed_levels[j], 
                                    self.embed_levels[j+1], self.f_size))

            self.decoder_param_list.append(temptemp) # Each level
            self.atn_list.append(self.self_atn_2d_def(self.embed_levels[j+1],self.embed_levels[j+1], 64))
            self.up_list.append(self.upconv_2d_def(self.embed_levels[j+1], 
                                    self.embed_levels[j+1]))
        self.decoder_param_list.append(self.conv2d_def(self.embed_levels[j+1], 
                                    self.embed_levels[j+1], self.f_size))
        ##################
        ##################

        #######################
        ## Prediction layers ##
        #######################
        self.s1=self.conv2d_def(self.embed_levels[j+1], 64, self.f_size)
        self.s2=self.conv2d_linear_def(64, self.num_classes, 1)
        ###############
        ###############
        self.myparameters = nn.ParameterList(self.params)
        print('MY PARAM LIST: ' + str(len(self.myparameters)))

        # Define normalization and place on device
        self.mu_tens=torch.tensor([[0.485, 0.456, 0.406]]).unsqueeze(-1).unsqueeze(-1).to(device)
        self.sig_tens=torch.tensor([[0.229, 0.224, 0.225]]).unsqueeze(-1).unsqueeze(-1).to(device)

        return





    def forward(self, x):

        x=self.conv_2d(x, self.pre_conv)

        x=self.pretrained_norm(x)

        o_size=x.size()
        x, intermed=self.encoder(x)

        x=self.ASPP_2d(x, self.bottle_neck)
        #x=self.self_atn_2d(x, self.bottle_neck_atn)  
        #x=self.ASPP_2d(x, self.bottle_neck2)  

        #
        #x=self.ASPP_2d(x, self.bottle_neck2)
        #x=self.ASPP_2d(x, self.bottle_neck3)

        x=self.decoder(x, intermed)     

        # Segmentation map
        s=self.conv_2d(x, self.s1) 
        s=self.linear_conv_2d(s, self.s2)  
        
        return s





    def encoder(self, x):

        connect_layers=[];
        connect_layers.append(x)

        #print(x.size())

        # Iterate through pretrained layers
        for ii, model in enumerate(self.res_new):
            x=model(x)

            # Keep track of ft maps used for skip connections
            if ii in self.drop_layers: 
                connect_layers.append(x)
        connect_layers.append(x)
        return x, connect_layers





    def decoder(self, x, connect_layers):

        # Iterate over each level of decoder
        for ii in range(len(self.up_list)):

            if False: #ii>1 and ii<(len(self.drop_layers)-2):
                conv_func=self.ASPP_2d
            else:
                conv_func=self.conv_2d 

            x=torch.cat((x,connect_layers[-1-ii]),axis=1)

            x=conv_func(x, self.decoder_param_list[ii][0])
            x=conv_func(x, self.decoder_param_list[ii][1])
            #x=self.self_atn_2d(x, self.atn_list[ii])    
            x=self.upconv_2d(x, self.up_list[ii], stri=2) # upsample by scale factor 2

        return self.conv_2d(x, self.decoder_param_list[-1])



    def pretrained_norm(self, x):
        # Required normalization for pretrained models
        #self.normalize = torchvision.transforms.Normalize
        #(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        #x_norm=self.normalize(x)

        x_norm=(x-self.mu_tens)/self.sig_tens

        return x_norm


    def conv2d_def(self, c_in, c_out, f_size=3):
        w= nn.Parameter(self.stvs*torch.randn(c_out, c_in, f_size, f_size)); # conv filter
        b= nn.Parameter(self.stvs*torch.ones(c_out)); # conv bias
        if self.norm==True:
            gamma= nn.Parameter(torch.ones(1,1,1,1)); # LN scale add c_out for IN
            beta= nn.Parameter( 0.01*torch.ones(1,1,1,1)); # LN shift
        alpha= nn.Parameter(torch.randn(c_out)); # PRELU parameter
        
        self.params.append(w)
        self.params.append(b)
        if self.norm==True:
            self.params.append(gamma)
            self.params.append(beta)
        self.params.append(alpha)
        if self.norm==True:
            return (w, b, gamma, beta, alpha)
        else:
            return (w, b, alpha)        
        
       
    def conv2d_linear_def(self, c_in, c_out, f_size=3):
        w= nn.Parameter(self.stvs*torch.randn(c_out, c_in, f_size, f_size)); # conv filter
        b= nn.Parameter(self.stvs*torch.ones(c_out)); # conv bias
        
        self.params.append(w)
        self.params.append(b)
        return (w, b)
        
     
    def conv_2d(self, x, conv, f_size=3, stri=1, dil=1):
        pc1=x.size(2)#.numpy()
        pc2=x.size(3)#.numpy()

        pad1=int(((pc1-1)*stri-pc1+f_size+(f_size-1)*(dil-1))/2)
        pad2=int(((pc2-1)*stri-pc2+f_size+(f_size-1)*(dil-1))/2)
        
        c=F.conv2d(x, conv[0], bias=conv[1], stride=stri, dilation=dil, padding=(pad1, pad2))
        
        if self.norm==True:
            c=self.LN_2d(c, conv[2], conv[3])
            p=self.PRELU_2d(c, conv[4])
        else:
            p=self.PRELU_2d(c, conv[2])
        return p
        
        
    def linear_conv_2d(self, x, conv, f_size=3, stri=1, dil=1):
        pc1=x.size(2)#.numpy()
        pc2=x.size(3)#.numpy()
        f_size_x=conv[0].size(-1)
        f_size_y=conv[0].size(-2)

        pad1=int(((pc1-1)*stri-pc1+f_size_y+(f_size_y-1)*(dil-1))/2)
        pad2=int(((pc2-1)*stri-pc2+f_size_x+(f_size_x-1)*(dil-1))/2)
        
        x=nn.ReflectionPad2d((pad2, pad2, pad1, pad1))(x)
        c=F.conv2d(x, conv[0], bias=conv[1], stride=stri, dilation=dil)
        
        return c

         
    def upconv_2d_def(self, c_in, c_out, f_size=3):
        m = nn.ConvTranspose2d(c_in, c_out, 3, stride=2, padding=1).to(self.device)
        gamma= nn.Parameter(torch.ones(1,c_out,1,1)); # LN scale
        beta= nn.Parameter( 0.01*torch.ones(1,c_out,1,1)); # LN shift
        self.params.append(gamma)
        self.params.append(beta)
        return (m, gamma, beta)


    def upconv_2d(self, x, params, f_size=3, stri=1, dil=1):
        y=params[0](x, output_size=(x.size(0), x.size(1), 2*x.size(2), 2*x.size(3)))
        y=self.LN_2d(y, params[1], params[2])
        y=nn.ReLU()(y)
        return y


        
    def LN_2d(self, x, gamma, beta):
        y=(x-torch.mean(x, dim=(1,2,3), keepdim=True))*gamma/(torch.std(x, dim=(1,2,3), keepdim=True)+self.const)+beta
        return y
        
        
    def PRELU_2d(self, x, alph):
        y=F.prelu(x, alph)
        return y
        
    def maxpool_2d(self, x):
        y = nn.MaxPool2d(3, stride=2)(x)
        return y
        
    def dropout(self, x, training=False, p=0.3):
        y=F.dropout(x, p=p, training=training)
        return y
       
    def upsample_2d(self, x, scale=2, interp=1):
        if interp==0:
            y=F.interpolate(x, scale_factor=scale, mode='nearest')
        else:
            y=F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=True)
        return y
        

    def resize_2d(self, x, h, w, interp=1):
        if interp==0:
            y=F.interpolate(x, size=(h,w), mode='nearest')
        else:
            y=F.interpolate(x, size=(h,w), mode='bilinear', align_corners=True)
        return y
        
    def softmax_2d(self, x):
        y=F.softmax(x, dim=1)
        return y




    def ASPP_2d_def(self, c_in, c_out, f_size=3):

        c1=self.conv2d_linear_def(c_in, int(c_out/4), f_size=f_size)
        c2=self.conv2d_linear_def(c_in, int(c_out/4), f_size=f_size)
        c3=self.conv2d_linear_def(c_in, int(c_out/4), f_size=f_size)
        c4=self.conv2d_linear_def(c_in, int(c_out/4), f_size=f_size)
        
        c5=self.conv2d_def(int(c_out/4)*4, c_out, f_size=f_size)
        
        return (c1, c2, c3, c4, c5)



    def ASPP_2d(self, x, params, f_size=3, stri=1):
        pc1=x.size(2)
        pc2=x.size(3)
        c1, c2, c3, c4, c5 = params

        x1=self.atrous_conv_2d(x, c1, 1, stri, f_size, pc1, pc2)
        x2=self.atrous_conv_2d(x, c1, 2, stri, f_size, pc1, pc2)
        x3=self.atrous_conv_2d(x, c1, 4, stri, f_size, pc1, pc2)
        x4=self.atrous_conv_2d(x, c1, 8, stri, f_size, pc1, pc2)
        
        conc=torch.cat((x1,x2,x3,x4), axis=1)
        y=self.conv_2d(conc, c5)

        return y


    def atrous_conv_2d(self, x, params, dil, stri, f_size, pc1, pc2):
        
        w,b=params  
        pad1=int(((pc1-1)*stri-pc1+f_size+(f_size-1)*(dil-1))/2)
        pad2=int(((pc2-1)*stri-pc2+f_size+(f_size-1)*(dil-1))/2)

        y=F.conv2d(x, w, bias=b, stride=stri, dilation=dil, padding=(pad1, pad2))
        return y



    def self_atn_2d_def(self, c_in, c_out, inter_dim):
        wg=nn.Parameter(self.stvs*torch.randn(c_in, inter_dim));
        wf=nn.Parameter(self.stvs*torch.randn(c_in, inter_dim));
        wh=nn.Parameter(self.stvs*torch.randn(c_in, inter_dim));
        wv=nn.Parameter(self.stvs*torch.randn(inter_dim, c_out));
        scale=nn.Parameter(torch.zeros(1));

        self.params.append(wg)
        self.params.append(wf)
        self.params.append(wh)
        self.params.append(wv)
        self.params.append(scale)
        
        return (wg, wf, wh, wv, scale)





    def self_atn_2d(self, x, params):
        wg, wf, wh, wv, scale=params
        c=x.size(1)
        h=x.size(2)
        w=x.size(3)

        wg=wg.unsqueeze(0).repeat(x.size(0),1,1) # Unsqueeze to batch dimension and replicate
        wf=wf.unsqueeze(0).repeat(x.size(0),1,1)
        wh=wh.unsqueeze(0).repeat(x.size(0),1,1)
        wv=wv.unsqueeze(0).repeat(x.size(0),1,1)

        x_squeeze=x.view(-1, c, h*w) # [b,c,H*W]
        #print(x_squeeze.size())

        fx=torch.bmm(torch.transpose(x_squeeze, self.dim1, self.dim2), wf) # [b, H*W, c_inter]
        gx=torch.bmm(torch.transpose(x_squeeze, self.dim1, self.dim2), wg) # [b, H*W, c_inter]
        inner=torch.exp(torch.bmm(fx, torch.transpose(gx, self.dim1, self.dim2)))  # [b, H*W, H*W]
        #print(fx.size(), gx.size(), inner.size())

        # Normalize down columns
        norm=inner/(torch.sum(inner, dim=1, keepdim=True) + self.const) # [b, H*W, H*W] -- Columns are normalized
        #print(norm.size())

        fh=torch.bmm(torch.transpose(x_squeeze, self.dim1, self.dim2), wh) # [b, H*W, c_inter]
        fh_t=torch.transpose(fh, self.dim1, self.dim2) # [b, c_inter, H*W]
        # print(fh.size(), fh_t.size())

        temp=torch.bmm(fh_t, norm)  # [b, c_inter, H*W]
        temp_t=torch.transpose(temp, self.dim1, self.dim2) # [b, H*W, c_inter]                
        out=torch.transpose(torch.bmm(temp_t, wv), self.dim1, self.dim2).unsqueeze(-1)  # [b, H*W, C_out] --> [b, c_out, H*W]

        #print(temp.size(), temp_t.size(), out.size())
        #print(out.size())
        #print(torch.reshape(out, (-1, out.size(1), h, w)).size())

        return torch.reshape(out, (-1, out.size(1), h, w))*scale+x






