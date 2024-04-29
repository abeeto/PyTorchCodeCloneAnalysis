import torch
import os

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from Data.Dataset import SegmentationDataset
import time
import segmentation_models_pytorch as smp

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    print('Running on the GPU')
else:
    DEVICE = "cpu"
    print('Running on the CPU')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGE_PATH="F:\\Poles\\Dataset\\Image2\\"
MASK_PATH="F:\\Poles\\Dataset\\Combine2\\"
TRAIN_RATIO=0.8
EPOCHS=20
LEARNING_RATE = 0.001
INPUT_IMAGE_HEIGHT=512
INPUT_IMAGE_WIDTH=512
files=os.listdir(IMAGE_PATH)
transforms = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((INPUT_IMAGE_HEIGHT,INPUT_IMAGE_WIDTH)),
	                            ])
train_size=int(TRAIN_RATIO*len(files))

train_dataset=SegmentationDataset(IMAGE_PATH, MASK_PATH, files[0:train_size], transforms)
trainLoader=DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataset=SegmentationDataset(IMAGE_PATH, MASK_PATH, files[train_size:], transforms)
testLoader=DataLoader(test_dataset, batch_size=1, shuffle=False)

# model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
#     in_channels=3, out_channels=1, init_features=32, pretrained=True)
model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=10,                      # model output channels (number of classes in your dataset)
)
if torch.cuda.is_available():
	model.cuda()
lossFunc = smp.losses.JaccardLoss(mode='multilabel')
opt = Adam(model.parameters(), lr=LEARNING_RATE)

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(EPOCHS)):
	# set the model in training mode
	model.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalTestLoss = 0
	# loop over the training set
	for (i, (x, y)) in enumerate(trainLoader):
		# send the input to the device
		(x, y) = (x.to(DEVICE), y.to(DEVICE))

		# perform a forward pass and calculate the training loss
		pred = model(x)
		loss = lossFunc(pred, y)
		print("[Train] {}/{}, Loss:{:.3f}".format(i,len(trainLoader), loss))
		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()
		# add the loss to the total training loss so far
		totalTrainLoss += loss
	print("Train loss: {:.6f}".format(
		totalTrainLoss/len(train_dataset)))
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()
		# loop over the validation set
		for (i, (x, y)) in enumerate(testLoader):
			# send the input to the device
			(x, y) = (x.to(DEVICE), y.to(DEVICE))
			# make the predictions and calculate the validation loss
			pred = model(x)
			totalTestLoss += lossFunc(pred, y)
			print("[Val] {}/{}".format(i, len(testLoader)))

		print("Test loss avg: {:0.6f}".format(totalTestLoss/len(testLoader)))
	torch.save(model.state_dict(), os.path.join("checkpoint/", 'checkpoint_sm_' + str(e) + '.zip'))

# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))