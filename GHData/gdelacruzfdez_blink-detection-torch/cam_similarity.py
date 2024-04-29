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



directories = [# "/mnt/hdd/gcruz/eyesOriginalSize2/RN15Train",
                "/mnt/hdd/gcruz/eyesOriginalSize2/RN30Train"]

train_transform = transforms.Compose([
    transforms.Resize((100, 100), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225])
])


class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features

    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(model_output, self.features)

model = network.EyeLRCN()
model.siamese.load_state_dict(torch.load('/home/gcruz/hdd/paper_models/eval_params/blink_detection/models/cnn/model.pt'))
model.lstm.load_state_dict(torch.load('/home/gcruz/hdd/paper_models/eval_params/blink_detection/models/lstm/model.pt'))
train_set = dataloader.BlinkDetectionLSTMDataset(directories, train_transform)
train_loader = DataLoader(
    train_set, batch_size=512, shuffle=False, num_workers=8)
target_layers = [model.siamese.embedding_net.model.layer4[-1]]
#torch.backends.cudnn.enabled=False

cam = EigenCAM(model=model.siamese.embedding_net, target_layers=target_layers, use_cuda=True)
progress = tqdm(enumerate(train_loader), total=len(
    train_loader), desc='Training', file=sys.stdout)
image_number = 0

for batch_idx, data in progress:
    samples, targets = data
    targets = [ClassifierOutputTarget(1)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=samples, targets=targets, aug_smooth= true, eigen_smooth=True)

    #eye_images = np.transpose(samples[0, :].numpy(), (1, 2,0))
    eye_images = np.transpose(samples.numpy(), (0, 2, 3, 1))
    for i in range(eye_images.shape[0]):
        visualization = show_cam_on_image(eye_images[i, :], grayscale_cam[i, :], use_rgb=True)
        f = plt.figure()
        f.add_subplot(1, 2, 1)
        plt.imshow(visualization)
        f.add_subplot(1, 2, 2)
        plt.imshow(eye_images[i, :])
        plt.savefig('/mnt/hdd/gcruz/cam/open/' + str(image_number) + '.png')
        #plt.show()
        plt.close(f)
        image_number += 1
