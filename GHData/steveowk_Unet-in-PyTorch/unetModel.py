import torch
import torch.nn as nn

# double convolution
def d_convd(in_channel, out_channel):
	double_conv_layer = nn.Sequential(
	                    nn.Conv2d(in_channel, out_channel,kernel_size=3),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channel, out_channel,kernel_size=3),
                        nn.ReLU(inplace=True)
		)
	return double_conv_layer

#cropping function
def crop_tensor(source_tensor, target_tensor):
		target_size = target_tensor.size()[2]
		source_size = source_tensor.size()[2]
		size_difference = (source_size-target_size)//2		
		res = source_tensor[:,:,size_difference:source_size-size_difference,size_difference:source_size-size_difference]
		return res

#deconvolution
def tranposed_conv(in_channel,out_channel):
	 return nn.ConvTranspose2d(in_channel,out_channel,2,stride=2)

# model
class Unet(nn.Module):
	def __init__(self):
		super(Unet, self).__init__()
		
		# pooling
		self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)

		# down convolution
		self.conv1 = d_convd(1,64)
		self.conv2 = d_convd(64,128)
		self.conv3 = d_convd(128,256)
		self.conv4 = d_convd(256,512)
		self.conv5 = d_convd(512,1024)

		# deconvolution layer
		self.transposed_conv1 = tranposed_conv(1024,512)
		self.transposed_conv2 = tranposed_conv(512,256)
		self.transposed_conv3 = tranposed_conv(256,128)
		self.transposed_conv4 = tranposed_conv(128,64)
		
		#up convolution
		self.dconv1 = d_convd(1024,512)
		self.dconv2 = d_convd(512,256)
		self.dconv3 = d_convd(256,128)
		self.dconv4 = d_convd(128,64)

		#output layer	
		self.out = nn.Conv2d(64,2,1)	
	
	def forward (self,image): 		
		#encoder
		in1 = self.conv1(image)		
		in1_pool = self.max_pool(in1)	
		in2 = self.conv2(in1_pool)
		in2_pool = self.max_pool(in2)	
		in3 = self.conv3(in2_pool)
		in3_pool = self.max_pool(in3)	
		in4 = self.conv4(in3_pool)
		in4_pool = self.max_pool(in4)	
		in5 = self.conv5(in4_pool)	
		
		# decoder 
		transposed_conv1 = self.transposed_conv1(in5)		
		cropped_tensor1 = crop_tensor(in4,transposed_conv1)			
		concat1 = torch.cat([cropped_tensor1,transposed_conv1],1)		
		conv_concat1 = self.dconv1(concat1)		

		transposed_conv2 = self.transposed_conv2(conv_concat1)		
		cropped_tensor2 = crop_tensor(in3,transposed_conv2)	
		concat2 = torch.cat([cropped_tensor2,transposed_conv2],1)		
		conv_concat2 = self.dconv2(concat2)	

		transposed_conv3 = self.transposed_conv3(conv_concat2)
		cropped_tensor3 = crop_tensor(in2,transposed_conv3)	
		concat3 = torch.cat([cropped_tensor3,transposed_conv3],1)
		conv_concat3 = self.dconv3(concat3)		

		transposed_conv4 = self.transposed_conv4(conv_concat3)
		cropped_tensor4 = crop_tensor(in1,transposed_conv4)	
		concat4 = torch.cat([cropped_tensor4,transposed_conv4],1)
		conv_concat4 = self.dconv4(concat4)		

		return self.out(conv_concat4)		


if __name__ == "__main__":
	image = torch.rand((1,1,572,572))
	model = Unet()
	out = model(image)
	print(out.shape)