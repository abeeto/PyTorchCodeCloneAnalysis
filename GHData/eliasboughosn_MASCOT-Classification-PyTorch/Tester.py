import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import multiprocessing
from sklearn.metrics import confusion_matrix
import torchvision.transforms.functional as F
# Paths for image directory and model
EVAL_DIR='/data/stars/user/eboughos/sex/all/combined/test'
EVAL_MODEL='../sex_images/ResNet18.pth'

# Load the model for evaluation
model = torch.load(EVAL_MODEL)
model.eval()

# Configure batch size and nuber of cpu's
num_cpu = multiprocessing.cpu_count()
bs = 8
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')



# Prepare the eval data loader
# eval_transform=transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])])
eval_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

eval_dataset=datasets.ImageFolder(root=EVAL_DIR, transform=eval_transform)
eval_loader=data.DataLoader(eval_dataset, batch_size=bs, shuffle=True,
                            num_workers=num_cpu, pin_memory=True)

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Number of classes and dataset-size
num_classes=len(eval_dataset.classes)
dsize=len(eval_dataset)

# Class label names
# class_names=['ACJY', 'E1', 'ISA3080', 'PJ', 'PMbio1', 'PR']
class_names=['male', 'female']

# Initialize the prediction and label lists
predlist=torch.zeros(0,dtype=torch.long, device='cpu')
lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

# Evaluate the model accuracy on the dataset
correct = 0
total = 0
with torch.no_grad():
    for images, labels in eval_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        predlist=torch.cat([predlist,predicted.view(-1).cpu()])
        lbllist=torch.cat([lbllist,labels.view(-1).cpu()])

# Overall accuracy
overall_accuracy=100 * correct / total
print('Accuracy of the network on the {:d} test images: {:.2f}%'.format(dsize,
    overall_accuracy))

# Confusion matrix
conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
print('Confusion Matrix')
print('-'*16)
print(conf_mat,'\n')

# Per-class accuracy
class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
print('Per class accuracy')
print('-'*18)
for label,accuracy in zip(eval_dataset.classes, class_accuracy):
     class_name=label
     print('Accuracy of class %8s : %0.2f %%'%(class_name, accuracy))

'''
Sample run: python eval.py eval_ds
'''
#Tracker and Detector code
import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import h5py
import os
import os.path as osp
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn



palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        if identities is not None:
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 +
                                     t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def process_dets(img, frame_idx, bboxes, identities, hf):
    group = hf.create_group(f'{frame_idx}')
    for i, box in enumerate(bboxes):
        side_long = int(max(box[2] - box[0], box[3] - box[1]) * 1.5)  # bbox inflated
        cr = (box[1] + box[3]) // 2
        cc = (box[0] + box[2]) // 2
        img_cropped = img[cr - side_long // 2:cr + side_long // 2,
                          cc - side_long // 2:cc + side_long // 2]
        group.create_dataset(f'{int(identities[i])}', data=img_cropped)


def detect(opt, save_img=False):
    path_save, source, weights, view_img, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Handling save locations
    if opt.name_vid != '':
        name_vid = opt.name_vid
    else:
        name_vid = osp.basename(os.path.normpath(source))[:-4]

    path_tracker_results = osp.join(path_save, 'Tracker_Results', name_vid)
    if osp.exists(path_tracker_results) and (opt.save_vid or opt.save_txt):
        shutil.rmtree(path_tracker_results)
    os.mkdir(path_tracker_results)
    path_save_vid = osp.join(path_tracker_results, name_vid+'.avi')
    if opt.save_det_vid:
        path_save_vid_det = osp.join(path_tracker_results, name_vid+'_det.avi')
    path_save_txt = osp.join(path_tracker_results, name_vid+'.txt')

    path_dets = osp.join(path_save, 'ReID_Dataset', name_vid+'.h5')
    if osp.exists(path_dets) and opt.save_det:
        os.remove(path_dets)
    if opt.save_det:
        hf_dets = h5py.File(path_dets, 'w')

    # Load model
    model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    vid_path_det, vid_writer_det = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Detection Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)

                if opt.save_det_vid:
                    img_det = im0.copy()
                    draw_boxes(img_det, det[:, :4].cpu().numpy().astype(int))
                    cv2.putText(img_det, str(frame_idx), (im0.shape[1] - 300, 100),
                                cv2.FONT_HERSHEY_PLAIN, 5, [2, 2, 2], 5)

                # draw boxes for visualization and/or process patches
                if len(outputs) > 0 and (opt.save_vid or opt.save_det):
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]

                    if opt.save_det:
                        process_dets(im0, frame_idx, bbox_xyxy, identities, hf_dets)
                    if opt.save_vid:
                        draw_boxes(im0, bbox_xyxy, identities)


                cv2.putText(im0, str(frame_idx), (im0.shape[1]-300, 100), cv2.FONT_HERSHEY_PLAIN, 5, [2, 2, 2], 5)

                # Write MOT compliant results to file
                if opt.save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(path_save_txt, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
            else:
                deepsort.increment_ages()
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (video)
            if opt.save_vid:
                if vid_path is None:  # new video
                    vid_path = path_save_vid
                    if opt.save_det_vid:
                        vid_path_det = path_save_vid_det
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if isinstance(vid_writer_det, cv2.VideoWriter) and opt.save_det_vid:
                        vid_writer_det.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(fps, w, h)
                    vid_writer = cv2.VideoWriter(
                        vid_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    if opt.save_det_vid:
                        vid_writer_det = cv2.VideoWriter(
                                        vid_path_det, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                vid_writer.write(im0)
                if opt.save_det_vid:
                    vid_writer_det.write(img_det)

    if vid_writer is not None:
        print('vid_path:', vid_path)
        vid_writer.release()

    if vid_writer_det is not None:
        vid_writer_det.release()

    if opt.save_txt or opt.save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + path_save)
        if platform == 'darwin':  # MacOS
            os.system('open ' + path_save)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/yolov5s.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='/data/stars/user/vpani/MASCOT',
                        help='output folder')  # output folder
    parser.add_argument('--output_video', type=str, default='Tracker_Results'),
    parser.add_argument('--output_reid', type=str, default='Tracker_Results'),
    parser.add_argument('--name_vid', type=str, default=''),
    parser.add_argument('--img-size', type=int, default=1280,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-vid', action='store_true',
                        help='save results to *.avi')
    parser.add_argument('--save-det', action='store_true',
                        help='save results to *.h5py')
    parser.add_argument('--save-det-vid', action='store_true',
                        help='save results to *.h5py')
    parser.add_argument('-v', '--video', action='store_true',
                        help='input video or not')
    parser.add_argument('--no-video', action='store_false',
                        help='save video or not')
    parser.add_argument('--save-crops', action='store_true',
                        help='save crops for reid dataset')
    # class 0 is PMbio1
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)
