import torch 
from torchvision import utils as tutils 
from tensorboardX import SummaryWriter 


writer = SummaryWriter()

image_black = torch.zeros(16,3,64,64)
image_white = torch.ones_like(image_black)


camaieu = torch.zeros(32,3,64,64)

for i in range(32): 

	if i%2 == 0: 
		camaieu[i,:,:,:] = image_black[int(i/2),:,:,:]
	else: 
		camaieu[i,:,:,:] = image_white[int((i-1)/2),:,:,:]


grid = tutils.make_grid(camaieu)
grid_2 =tutils.make_grid(torch.cat([image_black, image_white],0))
grid_3 =tutils.make_grid(torch.cat([image_white, image_black],0))
writer.add_image('Image', grid, 1)
writer.add_image('Image', grid_2, 2)
writer.add_image('Image', grid_3, 3)


