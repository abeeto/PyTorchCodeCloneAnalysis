import argparse
import torch
from torchvision import transforms
from PIL import Image
from retinanet import model
from datasets.dataloader import Normalizer, Resizer, UnNormalizer, DATASET_CLASSES
import cv2
import numpy as np
import time

parser = argparse.ArgumentParser(
    description='Simple test script ')

parser.add_argument('--model', default='retinanet', choices=['retinanet', 'centernet', 'faster-r-cnn'],
                    type=str, help='Choose model type')

parser.add_argument('--weight', type=str, help='Weight path')

parser.add_argument('--num_classes', default=3,
                    type=int, help='Number of classes (if not in checkpoint)')

parser.add_argument('--image', type=str, help='Image path')

# Retina only
parser.add_argument(
    '--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def read_image():
    # Read image and preprocess it
    img = Image.open(args.image).convert('RGB')
    print('Ori: ', img.size)
    data = {'img': np.array(
        img)/255.00, 'annot': np.array([[0.00, 0.00, 0.00, 0.00, 0.00, ]])}
    transform = transforms.Compose([Normalizer(), Resizer()])
    data = transform(data)
    return data['img'].permute(2, 0, 1).unsqueeze(0)


def read_checkpoint():
    # Read checkpoint to get info
    if args.weight is not None:
        checkpoint = torch.load(args.weight)
        if 'num_classes' not in checkpoint:
            checkpoint['num_classes'] = args.num_classes
    else:
        raise AttributeError('No weight provide')
    return checkpoint


def draw_caption(image, box, caption):

    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 2),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 2),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def demo_retina():
    checkpoint = read_checkpoint()
    # Create the model
    if args.depth == 18:
        retinanet = model.retina_bb_resnet18(
            num_classes=checkpoint['num_classes'], pretrained=True)
    elif args.depth == 34:
        retinanet = model.retina_bb_resnet34(
            num_classes=checkpoint['num_classes'], pretrained=True)
    elif args.depth == 50:
        retinanet = model.retina_bb_resnet50(
            num_classes=checkpoint['num_classes'], pretrained=True)
    elif args.depth == 101:
        retinanet = model.retina_bb_resnet101(
            num_classes=checkpoint['num_classes'], pretrained=True)
    elif args.depth == 152:
        retinanet = model.retina_bb_resnet152(
            num_classes=checkpoint['num_classes'], pretrained=True)
    else:
        raise ValueError(
            'Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    retinanet.load_state_dict(checkpoint['state_dict'])
    retinanet.to(device)
    retinanet.eval()
    retinanet.training = False
    unnormalize = UnNormalizer()
    image = read_image()

    with torch.no_grad():
        st = time.time()
        scores, classification, transformed_anchors = retinanet(
            image.to(device).float())
        print('Elapsed time: {}'.format(time.time()-st))
        idxs = np.where(scores.cpu() > 0.5)
        img = np.array(255 * unnormalize(image[0, :, :, :])).copy()

        img[img < 0] = 0
        img[img > 255] = 255

        img = np.transpose(img, (1, 2, 0))

        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            label_name = DATASET_CLASSES[int(classification[idxs[0][j]])]
            draw_caption(img, (x1, y1, x2, y2), label_name)

            cv2.rectangle(img, (x1, y1), (x2, y2),
                          color=(0, 0, 255), thickness=2)
            print(label_name)
        print(img.shape)
        cv2.imshow('img', img)
        cv2.waitKey(0)


if __name__ == "__main__":
    if args.model == 'retinanet':
        demo_retina()
