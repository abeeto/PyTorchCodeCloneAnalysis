import torch
import torchvision
import argparse
import cv2
from PIL import Image
import detect_utils


#memilih device gpu atau cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#mengatur inputan di terminal
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input image/video')
parser.add_argument('-m', '--min-size', dest='min_size', default=800,help='minimum input size for the FasterRCNN network')
args = vars(parser.parse_args())

#download model pretrained dari torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,min_size=args['min_size'])

# Eksekusi Model PreTrained 
image = Image.open(args['input'])
model.eval().to(device)
boxes, classes, labels = detect_utils.predict(image, model, device, 0.5)
image = detect_utils.draw_boxes(boxes, classes, labels, image)
counter = detect_utils.Counter(classes, nama_kelas = 'car')

# Menampilkan, Menyimpan Image 
cv2.imshow('Image', image)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}"
cv2.imwrite(f"outputs/{save_name}.jpg", image)
cv2.waitKey(0)
