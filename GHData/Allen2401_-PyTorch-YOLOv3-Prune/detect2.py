from __future__ import division

from DarknetModel import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from PIL import Image, ImageFont, ImageDraw

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from timeit import default_timer as timer

def detect_image(image,device):

    ## 图像处理
    ###
    start = timer()
    originshape = image.size[:2][::-1]
    img = transforms.ToTensor()(image)
    img, pad = pad_to_square(img, 0)
    _, padded_h, padded_w = img.shape
    img = resize(img,opt.img_size)
    ## 增加batch维度
    image_data = np.expand_dims(img, 0)
    image_data = torch.from_numpy(image_data).to(device)
    end1 = timer()
    print("handle:"+str(end1-start))
    ## 检测得到输出
    with torch.no_grad():
        detections = model(image_data)
        end2 = timer()
        print("infernce", str(end2 - end1))
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
    # end1 = timer()
    # print("infernce", str(end1 - start))
    classes = load_classes(opt.class_path)
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(1e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    # cmap = plt.get_cmap("tab20b")
    # colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    colors = ["blue","red"]
    print(colors)

    if detections[0] is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(detections[0], opt.img_size, originshape)
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
            label = '{} {:.2f}'.format(classes[int(cls_pred)], cls_conf.item())
            draw = ImageDraw.Draw(image)  # draw the original image
            label_size = draw.textsize(label, font)
            # y1 = max(0, np.floor(y1 + 0.5).astype('int32'))
            # x1 = max(0, np.floor(x1 + 0.5).astype('int32'))
            # y2 = min(image.size[1], np.floor(y2 + 0.5).astype('int32')
            # x2 = min(image.size[0], np.floor(x2 + 0.5).astype('int32'))
            if int(cls_pred)==1:
                continue
            y1 = max(0, np.floor(y1 + 0.5).int())
            x1 = max(0, np.floor(x1 + 0.5).int())
            y2 = min(image.size[1], np.floor(y2 + 0.5).int())
            x2 = min(image.size[0], np.floor(x2 + 0.5).int())
            print(label, (x1, y1), (x2, y2))
            box_w = x2 - x1
            box_h = y2 - y1
            if y1 - label_size[1] >= 0:
                text_origin = np.array([x1, y1 - label_size[1]])
            else:
                text_origin = np.array([x1, y1 + 1])
            # My kingdom for a good redistributable image drawing library.
            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # for i in range(thickness):
            draw.rectangle(
                [x1 + 10, y1 + 10, x2 - 10, y2 - 10],
                outline=colors[int(cls_pred)])
            ### this is label
            draw.rectangle(
                 [tuple(text_origin), tuple(text_origin + label_size)],
                 fill=colors[int(cls_pred)])
            draw.text(text_origin, label, fill='white', font=font)
            del draw
    end = timer()
    print("after handle"+str(end - end2))
    return image

def detect_video(video_path,device,output_path=""):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("couldn't open webcam r video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        frame = np.rot90(frame)
        image = Image.fromarray(frame)
        image = detect_image(image,device)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        # cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            print("-----------------------------------------------------------------------------")
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",type = bool,default=True,help="whether to detect video")
    parser.add_argument("--image_folder", type=str, default="data/sample", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/resnet_zh.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/resnet2/checkpoint_83.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/autodrive.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path)['state_dict'])

    model.eval()  # Set in evaluation mode
    if opt.video:
        detect_video("./data/test.mov",device,"./output/output.mov")
    else:
        dataloader = DataLoader(
            ImageFolder(opt.image_folder, img_size=opt.img_size),
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_cpu,
        )

        classes = load_classes(opt.class_path)  # Extracts class labels from file

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        imgs = []  # Stores image paths
        img_detections = []  # Stores detections for each image index

        print("\nPerforming object detection:")
        prev_time = time.time()
        for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
            # Configure input
            input_imgs = Variable(input_imgs.type(Tensor))

            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
                print(detections)

            # Log progress
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

            # Save image and detections
            imgs.extend(img_paths)
            img_detections.extend(detections)

        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        print("\nSaving images:")
        # Iterate through images and save plot of detections
        print(img_detections)
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

            print("(%d) Image: '%s'" % (img_i, path))

            # Create plot
            img = np.array(Image.open(path))
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            # Draw bounding boxes and labels of detections
            if detections is not None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )

            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            filename = path.split("/")[-1].split(".")[0]
            plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
            plt.close()

