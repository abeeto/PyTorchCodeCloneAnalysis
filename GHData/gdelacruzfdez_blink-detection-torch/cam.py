import torch
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import network
from PIL import Image
from torchvision.transforms import transforms
import dataloader
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt



directories = [#"/mnt/hdd/gcruz/eyesOriginalSize2/RTBENE_TRAIN"
    #"/mnt/hdd/gcruz/eyesOriginalSize2/RN15Train"
                "/mnt/hdd/gcruz/eyesOriginalSize2/eyeblink8"
               ]

train_transform = transforms.Compose([
    transforms.Resize((100, 100), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model = network.EyeLRCN()
model.siamese.load_state_dict(torch.load('/home/gcruz/hdd/paper_models/eval_params/blink_detection/models/cnn/model.pt'))
model.lstm.load_state_dict(torch.load('/home/gcruz/hdd/paper_models/eval_params/blink_detection/models/lstm/model.pt'))
train_set = dataloader.BlinkDetectionLSTMDataset(directories, train_transform)
train_loader = DataLoader(
    train_set, batch_size=256, shuffle=False, num_workers=4)
target_layers = [model.siamese.embedding_net.model.layer4[-1]]
#torch.backends.cudnn.enabled=False

cam = EigenCAM(model=model.siamese.embedding_net, target_layers=target_layers, use_cuda=True)
progress = tqdm(enumerate(train_loader), total=len(
    train_loader), desc='Training', file=sys.stdout)
image_number = 0

for batch_idx, data in progress:
    samples, targets_prev, not_transformed = data
    targets = [ClassifierOutputTarget(1)]
    samples = samples.cuda()

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=samples, targets=targets)

    #eye_images = np.transpose(samples[0, :].numpy(), (1, 2,0))
    eye_images = np.transpose(not_transformed.numpy(), (0, 2, 3, 1))
    for i in range(eye_images.shape[0]):
        visualization = show_cam_on_image(eye_images[i, :], grayscale_cam[i, :], use_rgb=True)
        blink_type = 'B' if targets_prev[i] == 1 else 'NB'
        img = plt.imshow(visualization)
        img.set_cmap('hot')
        plt.axis('off')
        plt.savefig('/mnt/hdd/gcruz/cam/eyeblink8/' + str(image_number) + '_CAM_' + blink_type +  '.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        img = plt.imshow(eye_images[i, :])
        img.set_cmap('hot')
        plt.axis('off')
        plt.savefig('/mnt/hdd/gcruz/cam/eyeblink8/' + str(image_number) + '_ORIGINAL_' + blink_type + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        image_number += 1

    # for i in range(eye_images.shape[0]):
    #     visualization = show_cam_on_image(eye_images[i, :], grayscale_cam[i, :], use_rgb=True)
    #     f = plt.figure()
    #     f.add_subplot(1, 2, 1)
    #     plt.imshow(visualization)
    #     plt.title('Blink' if targets_prev[i] == 1 else 'Not blink')
    #     f.add_subplot(1, 2, 2)
    #     plt.imshow(eye_images[i, :])
    #     blink_type = 'B' if targets_prev[i] == 1 else 'NB'
    #     plt.savefig('/mnt/hdd/gcruz/cam/RN30Train/' + str(image_number) + '_' + blink_type + '.png')
    #     #plt.show()
    #     plt.close(f)
    #     image_number += 1
