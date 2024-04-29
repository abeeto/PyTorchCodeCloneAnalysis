import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


def detect():
    img_size = opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    source, weights, half, view_img, fr_limit, save_folder = opt.source, opt.weights, opt.half, opt.view_img, opt.fr_limit, opt.save
    if save_folder is None:
        raise IOError('Please set save_folder')
    # Initialize device & model
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()
    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Set Dataloader
    dataset = LoadImages(source, img_size=img_size)

    # Run inference
    processed_time = 0
    t0 = time.time()

    for d_idx, (path, img, im0s, vid_cap) in enumerate(dataset):

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img[:, :, :, :])[0].float() if half else model(img[:, :, :, :])[0]


        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # t2 = torch_utils.time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, im0 = path, im0s

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                s = '%g: ' % d_idx
                s += '%gx%g ' % img.shape[2:]  # print string
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            # Stream results - if you wanna view the image, uncomment!
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
        t2 = torch_utils.time_synchronized()

        print('%sDone. (%.5fs)\n'%(s, t2-t1))

        if d_idx == fr_limit - 1: # --> if you wanna see only portion of images, Uncomment!
            processed_time = time.time() - t0
            break

    print('Done. (%.3fs)' % processed_time)
    print('%.2f FPS in %d frames' % (fr_limit/processed_time, fr_limit))
    if half:
        file_name = 'fps_half.txt'
    else:
        file_name = 'fps.txt'
    save_fps(save_folder,fps=fr_limit / processed_time , file_name=file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='input/cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='input/dataset/etri/etri.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='input/weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='../DL-DATASET/etri-safety_system/distort/videos/[Distort]_ETRI_Video_640x480.avi', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--fr-limit', type=int,default=3400, help='frame limit')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--save', type=str, help='save frame per second (FPS.txt)')

    opt = parser.parse_args()

    with torch.no_grad():
        detect()
