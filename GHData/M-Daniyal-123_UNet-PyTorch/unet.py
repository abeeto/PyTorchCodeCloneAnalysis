import torch
import torch.nn as nn


def conv_double(input_channels,output_channels):
  conv =  nn.Sequential(
      nn.Conv2d(input_channels,output_channels,kernel_size=3),
      nn.ReLU(inplace=True),
      nn.Conv2d(output_channels,output_channels,kernel_size=3),
      nn.ReLU(inplace=True)
  )
  return conv

def crop_image(image_tensor,target_tensor):
  target_size =  target_tensor.size()[2]
  tensor_size = image_tensor.size()[2]

  delta = tensor_size - target_size

  delta =  delta // 2

  return image_tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta]


class UNet(nn.Module):
  def __init__(self):
    super(UNet, self).__init__()

    ### Encoder 
    self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2,stride=2)
    self.down_conv_1 = conv_double(1, 64)
    self.down_conv_2 = conv_double(64,128)
    self.down_conv_3 = conv_double(128, 256)
    self.down_conv_4 = conv_double(256, 512)
    self.down_conv_5 = conv_double(512, 1024)
    

    ### Decoder
    self.up_trans_1 =  nn.ConvTranspose2d(in_channels= 1024,
                                          out_channels = 512,
                                          kernel_size=2,
                                          stride =2)
    self.up_conv_1 = conv_double(1024,512)


    self.up_trans_2 =  nn.ConvTranspose2d(in_channels= 512,
                                          out_channels = 256,
                                          kernel_size=2,
                                          stride =2)
    self.up_conv_2 = conv_double(512,256)


    self.up_trans_3 =  nn.ConvTranspose2d(in_channels= 256,
                                          out_channels = 128,
                                          kernel_size=2,
                                          stride =2)
    self.up_conv_3 = conv_double(256,128)


    self.up_trans_4 =  nn.ConvTranspose2d(in_channels= 128,
                                          out_channels = 64,
                                          kernel_size=2,
                                          stride =2)
    self.up_conv_4 = conv_double(128,64)



    self.out = nn.Conv2d(
        in_channels = 64,
        out_channels =2,
        kernel_size=1
    )

  

  def forward(self, image):
    """
    Image : size (Batch_Size,channels,572,572)
    """
    ### Expected size => batch_size,channel,height,weight
    ### Encoder ###
    
    var1 = self.down_conv_1(image)  
    var2= self.max_pool_2x2(var1)
    var3 = self.down_conv_2(var2)  
    var4= self.max_pool_2x2(var3)
    var5 = self.down_conv_3(var4)  
    var6= self.max_pool_2x2(var5)
    var7 = self.down_conv_4(var6) 
    var8= self.max_pool_2x2(var7)
    var9 = self.down_conv_5(var8)


    ### Decoder ### 
    var10 = self.up_trans_1(var9)
    temp_var7x10 = crop_image(var7,var10)
    var10 = self.up_conv_1(torch.cat([var10,temp_var7x10],1))

    var11 = self.up_trans_2(var10)
    temp_var5x11 = crop_image(var5,var11)
    var11 = self.up_conv_2(torch.cat([var11,temp_var5x11],1))

    var12 = self.up_trans_3(var11)
    temp_var3x12 = crop_image(var3,var12)
    var12 = self.up_conv_3(torch.cat([var12,temp_var3x12],1))

    var13 = self.up_trans_4(var12)
    temp_var1x13 = crop_image(var1,var13)
    var13 = self.up_conv_4(torch.cat([var13,temp_var1x13],1))

    x = self.out(var13)   ### Final Output ## Final Size (1,2,388,388)
                                              
    return x



if __name__ == "__main__":
    image = torch.rand((1,1,572,572))   ### It is just for checking the model
    model = UNet()

    print(model(image))  