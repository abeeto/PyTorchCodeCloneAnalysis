'''
Usgae:
python main.py 
--superres
'''

from pyexpat import model
import torch
import numpy as np
from PIL import Image
from model import PhysicalNN
import argparse
from torchvision import transforms
import datetime

import cv2
import Tools.Superres as Superres
import Tools.IOExt as IO
import Tools.Progressbar as progress

def imshow(img):
    # show UI
    cv2.imshow('Corrected', img)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    return key == ord("q")


def fromImage(dir, file_save_dir):
    # get input image
    img = Image.open(dir)
    # config output file name
    img_name = (dir.split('/')[-1]).split('.')[0]

    # Correction here
    corrected = core(img)
    # write output
    cv2.imwrite('{}/{}_corrected.jpg'.format(file_save_dir, img_name), corrected)
    # UI show
    # imshow(corrected)


def fromVideo(dir, file_save_dir):
    # get input stream
    cap = cv2.VideoCapture(dir)

    # read config
    video_name = (dir.split('/')[-1]).split('.')[0]
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * (sr.getScale() if useSuperres else 1)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * (sr.getScale() if useSuperres else 1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ready the first frame and start the loop
    success, image = cap.read()

    # select codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # create output stream
    out = cv2.VideoWriter(
        '{}/{}_corrected.mp4'.format(file_save_dir, video_name), fourcc, fps, (width, height))

    i = 1
    while success:
        # convert from CV2 image to PIL Image
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Correction here
        corrected = core(img)
        # write to stream
        out.write(corrected)
        # Check Next and get the image
        success, image = cap.read()
        # Log into progress bar
        progress.log(i, frame_count, '{}/{}'.format(i, frame_count))
        i+=1
        # UI Show
        if imshow(corrected): break

    # release!!!!!!
    out.release()


def core(img):
    # config output var
    corrected = img

    # core logic
    if useCorrection:
        inp = transform(corrected).unsqueeze(0)
        inp = inp.to(device)
        out = model(inp)

        # change it back to CV2 relayed format
        corrected = unloader(out.cpu().squeeze(0))

    corrected = cv2.cvtColor(np.array(corrected), cv2.COLOR_RGB2BGR)

    # Apply supe res if needed
    if useSuperres:
        corrected = sr.upsample(corrected)

    return corrected


def main(cpp, fp, mp, videoMode, correction, superres):
    # Check for GPU
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load cv2 super sres
    global sr
    sr = Superres.init(mp)

    # Load AI model
    global model
    model = PhysicalNN()
    model = torch.nn.DataParallel(model).to(device)
    checkpoint = torch.load(cpp, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.module
    model.eval()
    print("=> loaded model at epoch {}".format(checkpoint['epoch']))

    # init other global vars for core
    global transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    global unloader
    unloader = transforms.ToPILImage()
    global useCorrection
    useCorrection = correction
    global useSuperres
    useSuperres = superres

    # conifg output
    result_path = './data/output/{}x'.format((sr.getScale() if useSuperres else 1))
    video_result_path = IO.mkdir('{}/{}'.format(result_path, 'videos'))
    image_result_path = IO.mkdir('{}/{}'.format(result_path, 'images'))

    # Logic start
    starttime = datetime.datetime.now()
    dirs = IO.fileDirs(fp)
    for dir in dirs:
        if not videoMode:
            fromImage(dir, image_result_path)
        else:
            fromVideo(dir, video_result_path)
    endtime = datetime.datetime.now()
    print(' Done')

    # Calculate the time taken with this process
    print('Save at '+result_path)
    print('Cost {}s'.format(endtime-starttime))


if __name__ == '__main__':
    # input from user
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='path to the checkpoints tar',
                        default='./checkpoints/model_best_2842.pth.tar')
    parser.add_argument(
        '--images', help='path to the images folder', default='./data/input/images/')
    parser.add_argument('--videos', help='path to the videos folder',
                        default='./data/input/videos/')
    parser.add_argument("--model", default='./models/ESPCN_x2.pb',
                        help="path to the super resolution model")
    parser.add_argument('--correction', help='True/False',
                        default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--superres', help='True/False',
                        default=False, action=argparse.BooleanOptionalAction)

    # init configs
    args = parser.parse_args()
    modes = input('Video or Image?: ')
    videoMode = (modes.lower() == 'video')
    fp = args.videos if videoMode else args.images

    # start program
    main(args.checkpoint, fp, args.model,
         videoMode, args.correction, args.superres)
