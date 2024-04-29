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
    parser = argparse.ArgumentParser(description='YOLOv3Plus Detection')
    
    parser.add_argument('--trained_model', default='weights/',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--mode', default='image',
                        type=str, help='Use the data from image, video or camera')
    parser.add_argument('-size', '--input_size', default=416, type=int,
                        help='input_size')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')
    parser.add_argument('--conf_thresh', default=0.1, type=float,
                        help='Confidence threshold')
    parser.add_argument('--nms_thresh', default=0.50, type=float,
                        help='NMS threshold')
    parser.add_argument('--path_to_img', default='data/demo/images/',
                        type=str, help='The path to image files')
    parser.add_argument('--path_to_vid', default='data/demo/videos/',
                        type=str, help='The path to video files')
    parser.add_argument('--path_to_save', default='det_results/',
                        type=str, help='The path to save the detection results')
    parser.add_argument('-vs', '--visual_threshold', default=0.3,
                        type=float, help='visual threshold')
    
    return parser.parse_args()
                    


def vis(img, bboxs, scores, cls_inds, class_color, thresh=0.3):
        
    for i, box in enumerate(bboxs):
        if scores[i] > thresh:
            cls_indx = cls_inds[i]
            cls_id = coco_class_index[int(cls_indx)]
            cls_name = coco_class_labels[cls_id]
            mess = '%s: %.3f' % (cls_name, scores[i])
            # bounding box
            xmin, ymin, xmax, ymax = box
            box_w = int(xmax - xmin)
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[int(cls_indx)], 2)
            cv2.rectangle(img, (int(xmin), int(abs(ymin)-15)), (int(xmin+box_w*0.55), int(ymin)), class_color[int(cls_indx)], -1)

            cv2.putText(img, mess, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    return img


def detect(net, device, transform, thresh, mode='image', path_to_img=None, path_to_vid=None, path_to_save=None):
    class_color = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(80)]
    save_path = os.path.join(path_to_save, mode)
    os.makedirs(save_path, exist_ok=True)

    # ------------------------- Camera ----------------------------
    if mode == 'camera':
        print('use camera !!!')
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            h, w, _ = frame.shape
            size = np.array([[w, h, w, h]])

            if cv2.waitKey(1) == ord('q'):
                break

            # preprocess
            img, _, _, scale, offset = transform(frame)
            x = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1).float()
            x = x.unsqueeze(0).to(device)

            t0 = time.time()
            # forward
            bboxes, scores, cls_inds = net(x)
            print("detection time used ", time.time() - t0, "s")
            
            # map the boxes to original image
            bboxes -= offset
            bboxes /= scale
            bboxes *= size

            frame_processed = vis(img=frame, 
                                  bboxs=bboxes,
                                  scores=scores, 
                                  cls_inds=cls_inds,
                                  class_color=class_color,
                                  thresh=thresh
                                  )
            cv2.imshow('detection result', frame_processed)
            cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()

    # ------------------------- Image ----------------------------
    elif mode == 'image':
        for i, img_id in enumerate(os.listdir(path_to_img)):
            img_raw = cv2.imread(path_to_img + '/' + img_id, cv2.IMREAD_COLOR)
            h, w, _ = img_raw.shape
            size = np.array([[w, h, w, h]])

            # preprocess
            img, _, _, scale, offset = transform(img_raw)
            x = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1).float()
            x = x.unsqueeze(0).to(device)

            t0 = time.time()
            # forward
            bboxes, scores, cls_inds = net(x)
            print("detection time used ", time.time() - t0, "s")
            
            # map the boxes to original image
            bboxes -= offset
            bboxes /= scale
            bboxes *= size

            img_processed = vis(img=img_raw, 
                                bboxs=bboxes,
                                scores=scores, 
                                cls_inds=cls_inds,
                                class_color=class_color,
                                thresh=thresh
                                )
            cv2.imshow('detection', img_processed)
            cv2.imwrite(os.path.join(save_path, str(i).zfill(6)+'.jpg'), img_processed)
            cv2.waitKey(0)

    # ------------------------- Video ---------------------------
    elif mode == 'video':
        video = cv2.VideoCapture(path_to_vid)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_size = (640, 480)
        save_path = os.path.join(save_path, 'det.avi')
        fps = 15.0
        out = cv2.VideoWriter(save_path, fourcc, fps, save_size)

        while(True):
            ret, frame = video.read()
            
            if ret:
                # ------------------------- Detection ---------------------------
                h, w, _ = frame.shape
                size = np.array([[w, h, w, h]])

                if cv2.waitKey(1) == ord('q'):
                    break

                # preprocess
                img, _, _, scale, offset = transform(frame)
                x = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1).float()
                x = x.unsqueeze(0).to(device)

                t0 = time.time()
                # forward
                bboxes, scores, cls_inds = net(x)
                print("detection time used ", time.time() - t0, "s")
                
                # map the boxes to original image
                bboxes -= offset
                bboxes /= scale
                bboxes *= size
                
                frame_processed = vis(img=frame, 
                                      bboxs=bboxes,
                                      scores=scores, 
                                      cls_inds=cls_inds,
                                      class_color=class_color,
                                      thresh=thresh
                                      )
                frame_processed_resize = cv2.resize(frame_processed, save_size)
                out.write(frame_processed_resize)
                cv2.imshow('detection', frame_processed)
                cv2.waitKey(1)
            else:
                break
        video.release()
        out.release()
        cv2.destroyAllWindows()


def run():
    args = parse_args()

    # use cuda
    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # model
    model_name = args.version
    print('Model: ', model_name)

    # load model and config file
    if model_name == 'yolov3p_cd53':
        from models.yolo_v3_plus import YOLOv3Plus as yolov3p_net
        cfg = config.yolov3plus_cfg
        backbone = cfg['backbone']
        anchor_size = cfg['anchor_size']

    else:
        print('Unknown model name...')
        exit(0)

    # input size
    input_size = args.input_size

    class_colors = [(np.random.randint(255),
                    np.random.randint(255),
                    np.random.randint(255)) for _ in range(80)]

    # build model
    net = yolov3p_net(device=device, 
                        input_size=input_size, 
                        num_classes=80, 
                        trainable=False, 
                        anchor_size=anchor_size, 
                        bk=backbone
                        )

    # load weight
    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.to(device).eval()
    print('Finished loading model!')

    # run
    detect(net=net, 
            device=device,
            transform=BaseTransform(input_size),
            mode=args.mode,
            path_to_img=args.path_to_img,
            path_to_vid=args.path_to_vid,
            path_to_save=args.path_to_save,
            thresh=args.visual_threshold
            )


if __name__ == '__main__':
    run()
