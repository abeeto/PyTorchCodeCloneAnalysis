import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
multi_gpu = True
conf = {
    "epochs": 150,
    "learning_rate": 0.0004,
    "momentum": 0.9,
    "batch_size": 1,
    "weight_decay": 0.0001,
    "weight_balance": [1, 50],
    "backbone": "efficientnet-b1",
    "positional_encoding": True,
    "loss_function": "Focal"
}

model = smp.Unet(conf["backbone"], encoder_weights='imagenet', in_channels=5, classes=2,
                 activation='softmax2d')
state_dict = torch.load("wandb/run-20200528_011112-2v8ch06i/efficientnet-b1-12-0.9841508527842078.pth")
model = nn.DataParallel(model) # because we trained in multi-gpu
model.load_state_dict(state_dict)

model.module.encoder.set_swish(memory_efficient=False)
model.eval()
model = model.cpu()

testset = AudiSegmentationDataset("/mnt/sda/datasets/Audi/camera_lidar_semantic", learning_map, split="val",
                                  positional=conf["positional_encoding"])
testloader = torch.utils.data.DataLoader(testset, batch_size=conf["batch_size"], shuffle=False) # the testset should be sequential, so it can make a video

confidence_thresh = 0.7 # if we're less than this, consider background
for batch, (images, labels) in enumerate(testloader):
    yhat = model(images).numpy()
    roadline_prob = yhat[:, 1, :, :] # now we've got a 1 x width x height of the probability it's a roadline
    line_mask = roadline > confidence_thresh # now this is 1 if roadline, 0 otherwise
    rgb_mask = line_mask * 255
    camera_images = (images[0, :3, :, :] * 128) + 128  # from [-1, 1] to [0, 255]
    camera_images[2, :, :] = np.maximum(camera_images[2, :, :], rgb_mask)
    im = Image.fromarray(camera_images)
    im.save(f"test/{batch:07d}image.png") # 7 digits so they all come out in a good order
