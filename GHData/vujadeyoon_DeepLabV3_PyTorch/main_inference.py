import argparse
import numpy as np
import torch
# import PIL
from PIL import Image
from torchvision import transforms
import torchvision.models as models

parser = argparse.ArgumentParser(description='Deep Lab Ver3 for semantic segmentation')
parser.add_argument('--path_img', type=str, default='./test_images/img_input_1.jpg', help='The path of input rgb image')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() and opt.cuda else 'cpu')

preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _prepare_model(_pretrained=True):
    '''
    1. Original url:
        A. https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
        B. https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth
    2. Download path:
        A. /home/usrname/.cache/torch/hub/pytorch_vision_v0.5.0
        B. /home/usrname/.cache/torch/checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth
    '''

    if _pretrained:
        model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=True).eval()
        # model = models.segmentation.deeplabv3_resnet101(pretrained=_pretrained).eval().to(device)
    else:
        raise NotImplementedError

    return model


def _get_img(_path_img=opt.path_img, _preprocess=preprocess):
    input_image = Image.open(_path_img)

    input_tensor = _preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    return input_batch


def _run_model(_img_rgb):
    with torch.no_grad():
        output = model(_img_rgb)['out'][0]

    segmented_map = output.argmax(0)

    return segmented_map


def _colorization_segmented_map(_segmented_map, _img_rgb_size):
    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    segmented_map_color = Image.fromarray(_segmented_map.byte().cpu().numpy()).resize(_img_rgb_size)
    segmented_map_color.putpalette(colors)

    return segmented_map_color


if __name__ == "__main__":
    model = _prepare_model(_pretrained=True)
    img_rgb = _get_img(_path_img=opt.path_img, _preprocess=preprocess)
    segmented_map = _run_model(_img_rgb=img_rgb)
    segmented_map_color = _colorization_segmented_map(_segmented_map=segmented_map, _img_rgb_size=tuple([img_rgb.size(3), img_rgb.size(2)]))

    img_extension = opt.path_img[opt.path_img.rfind('.'):]
    segmented_map_color.save(opt.path_img.replace(img_extension, '_segmap.png'))
