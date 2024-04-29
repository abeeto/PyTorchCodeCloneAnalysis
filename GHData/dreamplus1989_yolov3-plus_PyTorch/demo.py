import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from data import *
import numpy as np
import cv2
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection')

    parser.add_argument('-v', '--version', default='yolo_v3_plus',
                        help='yolo_v3_plus, yolo_v3_spp, yolo_v3_slim')
    parser.add_argument('-d', '--dataset', default='COCO',
                        help='COCO dataset')
    parser.add_argument('--mode', default='image',
                        type=str, help='Use the data from dataset, image, video or camera')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Use cuda')
    parser.add_argument('-size', '--input_size', default=416, type=int, 
                        help='The input size of image')
    parser.add_argument('--trained_model', default='weights/coco/yolo_v3_plus/yolo_v3_plus_260_37.40_57.42.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--cam_ind', default=0, type=int,
                        help='0: laptop camera; 1: external USB camera')
    parser.add_argument('--path_to_img', default='data/demo/Images/',
                        type=str, help='The path to image files')
    parser.add_argument('--path_to_vid', default='data/demo/video/',
                        type=str, help='The path to video files')
    parser.add_argument('--path_to_save', default='test_results/',
                        type=str, help='The path to save the detection results video')
    parser.add_argument('--conf_thresh', default=0.1, type=float,
                        help='Confidence threshold')
    parser.add_argument('--nms_thresh', default=0.45, type=float,
                        help='NMS threshold')
    parser.add_argument('--diou_nms', action='store_true', default=False, 
                        help='use diou_nms.')
    parser.add_argument('-vs','--vis_thresh', default=0.4,
                        type=float, help='visual threshold')
    
    return parser.parse_args()
                    

def preprocess(img):
    h, w, c = img.shape
    # zero padding
    if h > w:
        img_ = np.zeros([h, h, 3])
        delta_w = h - w
        left = delta_w // 2
        img_[:, left:left+w, :] = img
        offset = np.array([[ left / h, 0.,  left / h, 0.]])

    elif h < w:
        img_ = np.zeros([w, w, 3])
        delta_h = w - h
        top = delta_h // 2
        img_[top:top+h, :, :] = img
        offset = np.array([[0.,    top / w, 0.,    top / w]])
    
    else:
        img_ = img
        offset = np.zeros([1, 4])

    return img_, offset, h, w


def vis(img, bbox_pred, scores, cls_inds, thresh, class_color, class_names=None):
    for i, box in enumerate(bbox_pred):
        if scores[i] > thresh:
            cls_indx = int(cls_inds[i])
            cls_id = coco_class_index[int(cls_indx)]
            cls_name = coco_class_labels[cls_id]
            mess = '%s: %.3f' % (cls_name, scores[i])
            # bounding box
            xmin, ymin, xmax, ymax = box
            # print(xmin, ymin, xmax, ymax)
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[cls_indx], 1)
            cv2.rectangle(img, (int(xmin), int(abs(ymin)-15)), (int(xmax), int(ymin)), class_color[cls_indx], -1)
            cv2.putText(img, mess, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return img


def detect(args, net, device, transform, mode='image', path_to_img=None, path_to_vid=None, path_to_save=None, thresh=None, testset=None, class_color=None):
    if path_to_save is not None and isinstance(path_to_save, str):
        os.makedirs(path_to_save, exist_ok=True)
    # ------------------------- Camera ----------------------------
    if mode == 'camera':
        print('use camera !!!')
        cap = cv2.VideoCapture(args.cam_ind, cv2.CAP_DSHOW)

        while True:
            ret, frame = cap.read()

            if cv2.waitKey(1) == ord('q'):
                exit(0)

            if ret:
                # preprocess
                frame_, offset, h, w = preprocess(frame)

                # to rgb
                x = torch.from_numpy(transform(frame_)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
                x = x.unsqueeze(0).to(device)

                t0 = time.time()
                bboxes, scores, cls_inds = net(x)
                t1 = time.time()
                print("detection time used ", t1-t0, "s")
                # scale each detection back up to the image
                max_line = max(h, w)
                # map the boxes to input image with zero padding
                bboxes *= max_line
                # map to the image without zero padding
                bboxes -= (offset * max_line)

                frame_processed = vis(frame, bboxes, scores, cls_inds, thresh, class_color=class_color)
                cv2.imshow('detection result', frame_processed)
                cv2.waitKey(1)
            else:
                break
        
        cap.release()
        cv2.destroyAllWindows()        

    # ------------------------- Image ----------------------------
    elif mode == 'image':
        save_path = 'test_results/Images/'
        os.makedirs(save_path, exist_ok=True)

        for index, file in enumerate(os.listdir(path_to_img)):
            img = cv2.imread(path_to_img + '/' + file, cv2.IMREAD_COLOR)

            # preprocess
            img_, offset, h, w = preprocess(img)

            # to rgb
            x = torch.from_numpy(transform(img_)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
            x = x.unsqueeze(0).to(device)

            t0 = time.time()
            bboxes, scores, cls_inds = net(x)
            t1 = time.time()
            print("detection time used ", t1-t0, "s")
            # scale each detection back up to the image
            max_line = max(h, w)
            # map the boxes to input image with zero padding
            bboxes *= max_line
            # map to the image without zero padding
            bboxes -= (offset * max_line)

            img_processed = vis(img, bboxes, scores, cls_inds, thresh=thresh, class_color=class_color)
            cv2.imwrite(os.path.join(save_path, str(index).zfill(6) +'.jpg'), img_processed)
            cv2.imshow('detection result', img_processed)
            cv2.waitKey(0)

    # ------------------------- Video ---------------------------
    elif mode == 'video':
        save_path = 'test_results/Videos/'
        video = cv2.VideoCapture(path_to_vid)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(save_path, 'output.avi'), fourcc, 30.0, (640, 360))        
        
        while(True):
            ret, frame = video.read()
            
            if ret:
                # ------------------------- Detection ---------------------------

                # preprocess
                frame_, offset, h, w = preprocess(frame)

                # to rgb
                x = torch.from_numpy(transform(frame_)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
                x = x.unsqueeze(0).to(device)

                t0 = time.time()
                bboxes, scores, cls_inds = net(x)
                t1 = time.time()
                print("detection time used ", t1-t0, "s")
                # scale each detection back up to the image
                max_line = max(h, w)
                # map the boxes to input image with zero padding
                bboxes *= max_line
                # map to the image without zero padding
                bboxes -= (offset * max_line)
                
                frame_processed = vis(frame, bboxes, scores, cls_inds, thresh, class_color=class_color)
                
                resize_frame_processed = cv2.resize(frame_processed, (640, 360))
                cv2.imshow('detection result', frame_processed)
                out.write(resize_frame_processed)
                cv2.waitKey(1)
            else:
                break
        video.release()
        out.release()
        cv2.destroyAllWindows()

    # ------------------------- Dataset ---------------------------
    elif mode == 'dataset':
        class_color = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(80)]
        num_images = len(testset)
        for index in range(num_images):
            print('Testing image {:d}/{:d}....'.format(index+1, num_images))
            if args.dataset == 'COCO':
                img, _ = testset.pull_image(index)
                img_tensor, _, h, w, offset, scale = testset.pull_item(index)
            else:
                print('We only support COCO dataset !!!')

            x = img_tensor.unsqueeze(0).to(device)

            t0 = time.time()
            bboxes, scores, cls_inds = net(x)
            t1 = time.time()
            print("detection time used ", t1-t0, "s")
            # scale each detection back up to the image
            max_line = max(h, w)
            # map the boxes to input image with zero padding
            bboxes *= max_line
            # map to the image without zero padding
            bboxes -= (offset * max_line)

            img_processed = vis(img, bboxes, scores, cls_inds, thresh, class_color=class_color)
            cv2.imshow('detection result', img_processed)
            cv2.waitKey(0)
            # cv2.imwrite(os.path.join(save_path, str(index).zfill(6) +'.jpg'), img_processed)


def run():
    args = parse_args()
    input_size = [args.input_size, args.input_size]
    class_color = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(80)]

    # cuda
    if args.no_cuda:
        print("use cpu")
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            print("use gpu")
            device = torch.device("cuda")
        else:
            print("It seems you don't have a gpu ... ")
            device = torch.device("cpu")

    # build the model
    if args.version == 'yolo_v3_plus':
        print('loading YOLOv3Plus ...')
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        net = YOLOv3Plus(device, num_classes=80, input_size=input_size, anchor_size=anchor_size, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, diou_nms=args.diou_nms)

    elif args.version == 'yolo_v3_spp':
        print('loading YOLOv3Spp ...')
        from models.yolo_v3_spp import YOLOv3SPP
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        net = YOLOv3SPP(device, num_classes=80, input_size=input_size, anchor_size=anchor_size, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, diou_nms=args.diou_nms)

    elif args.version == 'yolo_v3_slim':
        print('loading YOLOv3Slim ...')
        from models.yolo_v3_slim import YOLOv3Slim
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        net = YOLOv3Slim(device, num_classes=80, input_size=input_size, anchor_size=anchor_size, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, diou_nms=args.diou_nms)
   
    else:
        print("We only support yolo_v3_plus, yolo_v3_spp, yolo_v3_slim, yolo_v3_tiny !! Please verify whether the --version or -v you entered meets the requirements ")
        exit(0)

    # load a trained model
    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.to(device).eval()
    print('Finished loading model!')

    # run
    if args.mode == 'camera':
        detect(args=args, net=net, device=device, transform=BaseTransform(input_size), 
               mode=args.mode, thresh=args.vis_thresh, class_color=class_color)
    elif args.mode == 'image':
        detect(args=args, net=net, device=device, transform=BaseTransform(input_size), 
               mode=args.mode, thresh=args.vis_thresh, path_to_img=args.path_to_img, path_to_save=args.path_to_save, class_color=class_color)
    elif args.mode == 'video':
        detect(args=args, net=net, device=device, transform=BaseTransform(input_size),
               mode=args.mode, thresh=args.vis_thresh, path_to_vid=args.path_to_vid, path_to_save=args.path_to_save, class_color=class_color)
    elif args.mode == 'dataset':
        # build test dataset
        if args.dataset == 'COCO':
            testset = COCODataset(
                        data_dir=coco_root,
                        json_file='instances_val2017.json',
                        name='val2017',
                        img_size=input_size[0],
                        transform=BaseTransform(input_size),
                        debug=False)

        detect(args=args,
               net=net, 
               device=device,
               transform=BaseTransform(input_size),
               mode=args.mode, 
               thresh=args.vis_thresh, 
               path_to_img=args.path_to_img, 
               path_to_save=args.path_to_save,
               testset=testset,
               class_color=class_color)

    else:
        print("We only support camera, image, video and dataset !! Please verify whether the --mode you entered meets the requirements ")
        exit(0)

if __name__ == '__main__':
    run()