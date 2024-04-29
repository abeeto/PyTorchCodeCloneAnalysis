import os
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.backends import cudnn
from torch.utils.data import DataLoader

from torchvision import transforms

from core.resnet38_cls import Classifier

from tools.utils import *
from tools.txt_utils import *
from tools.dataset_utils import *
from tools.augment_utils import *
from tools.pickle_utils import *
from tools.torch_utils import *

###############################################################################
# Arguments
###############################################################################
parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--architecture', default='resnet38', type=str)

parser.add_argument('--la_crf', default=4, type=int)
parser.add_argument('--ha_crf', default=32, type=int)

parser.add_argument('--root_dir', default='F:/VOCtrainval_11-May-2012/', type=str)

parser.add_argument('--domain', default='val', type=str)
parser.add_argument('--model_name', default='VOC2012@arch=resnet38@pretrained=True@lr=0.1@wd=0.0005@bs=16@epoch=15', type=str)

args = parser.parse_args()
###############################################################################

if __name__ == '__main__':
    cam_dir = create_directory(f'./predictions/{args.model_name}/cam/')
    la_crf_dir = create_directory(f'./predictions/{args.model_name}/la_crf_{args.la_crf}/')
    ha_crf_dir = create_directory(f'./predictions/{args.model_name}/ha_crf_{args.ha_crf}/')

    model_dir = create_directory('./models/')
    model_path = model_dir + f'{args.model_name}.pth'

    cudnn.enabled = True

    test_transform = transforms.Compose([
        np.asarray,
        Normalize(),

        Transpose2D(),
        torch.from_numpy
    ])

    class_names = np.asarray(read_txt('./data/class_names.txt'))
    class_indices = np.arange(len(class_names))
    
    class_dic = {class_name:class_index for class_index, class_name in enumerate(class_names)}

    model = Classifier(len(class_names))

    dataset = VOC_Dataset_with_Mask(args.root_dir, f'./data/{args.domain}.txt', class_dic, None)
    
    model = model.cuda()
    model.eval()
    
    load_model(model, model_path, False)

    cmap_dic, _ = get_color_map_dic('PASCAL_VOC')
    colors = np.asarray([cmap_dic[class_name] for class_name in class_names])

    def get_cam(image, label, image_size):
        image = Image.fromarray(image)
        images = test_transform(image)
        images = images.unsqueeze(0)
        images = images.cuda()

        logits, features = model(images, with_cam=True)

        cams = F.upsample(features, image_size, mode='bilinear', align_corners=False)
        cams = get_numpy_from_tensor(cams[0]) * label.reshape((-1, 1, 1))
        return cams

    def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax

        h, w = img.shape[:2]
        n_labels = labels

        d = dcrf.DenseCRF2D(w, h, n_labels)

        unary = unary_from_softmax(probs)
        unary = np.ascontiguousarray(unary)

        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
        d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
        Q = d.inference(t)

        return np.array(Q).reshape((n_labels, h, w))

    def _crf_with_alpha(cam_dict, alpha):
        v = np.array(list(cam_dict.values()))
        
        bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
        bgcam_score = np.concatenate((bg_score, v), axis=0)

        crf_score = crf_inference(ori_image, bgcam_score, labels=bgcam_score.shape[0])
        
        n_crf_al = dict()

        n_crf_al[0] = crf_score[0]
        for i, key in enumerate(cam_dict.keys()):
            n_crf_al[key+1] = crf_score[i+1]

        return n_crf_al
        
    length = len(dataset)
    mIoU_list = []

    for i, (image_name, image, label, mask) in enumerate(dataset):
        sys.stdout.write('\r[{}/{}]'.format(i + 1, length))
        sys.stdout.flush()
        
        # calculate class activation maps
        cams_list = []

        ori_image = np.asarray(image)
        ori_image_size = ori_image.shape[:2]

        for scale in [1.0, 0.5, 1.5, 2.0]:
            for flip in [False, True]:
                image = cv2.resize(ori_image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

                if flip: image = image[:, ::-1, :]
                cams = get_cam(image, label, ori_image_size)
                if flip: cams = cams[:, :, ::-1]

                cams_list.append(cams)
        
        cams = np.sum(cams_list, axis=0)
        cams = denormalize_for_cam(cams)

        cam_dict = {class_index+1:cams[..., class_index] for class_index in class_indices[label==1.]}

        # 1. CAM
        dump_pickle(cam_dir + image_name, cam_dict)

        # 2. CAM - low alpha
        dump_pickle(la_crf_dir + image_name, _crf_with_alpha(cam_dict, args.la_crf))

        # 3. CAM - high alpha
        dump_pickle(ha_crf_dir + image_name, _crf_with_alpha(cam_dict, args.ha_crf))
        
        # bg = np.ones_like(cams[:, :, 0]) * 0.2
        # cams = np.argmax(np.concatenate([bg[..., np.newaxis], cams], axis=-1), axis=-1)
        
        # mIoU = get_meanIU(cams, gt)
        # mIoU_list.append(mIoU)
    
