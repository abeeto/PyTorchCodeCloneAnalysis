import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import Resizer, Normalizer
from retinanet.dataloader import AIZOODataset
from retinanet import coco_eval

assert torch.__version__.split('.')[0] == '1'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--model_path', help='Path to model', type=str)

    parser = parser.parse_args(args)

    model_path = './model_final.pt'
    test_path = './test'
    dataset_test = AIZOODataset(test_path, transforms=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    retinanet = model.resnet50(num_classes=3, pretrained=False)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.load(model_path)
        #retinanet.load_state_dict(checkpoint.module)
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    #retinanet.freeze_bn()

    coco_eval.evaluate_coco(dataset_test, retinanet)


if __name__ == '__main__':
    main()
