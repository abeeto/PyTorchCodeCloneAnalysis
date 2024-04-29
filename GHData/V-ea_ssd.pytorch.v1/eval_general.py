from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
# from data import VOC_CLASSES as labelmap
import torch.utils.data as data
from data.SIXrayDetectionEval import MY_CLASSES as labelmap, SIXrayDetectionEval, SIXrayAnnotationTransform

from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/SIXRAY3.pth', type=str,
                    # default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
# parser.add_argument('--imagesetfile', default=None,
#                     # parser.add_argument('--voc_root', default=VOC_ROOT,
#                     help='测试文件的输入包含一个txt文件（里面全是测试集合的图片名），为None时就验证所有imgpath中的图片')
# parser.add_argument('--imgpath', default="data/SIXRay_test/images/",
#                     help='测试图片所在的文件夹路径')
# parser.add_argument('--annopath', default="data/SIXRay_test/anno/",
#                     help='测试图片的标注文件所在的文件夹路径')
parser.add_argument('--result_file', default="data/SIXRay_test/result.txt",
                    help='最终结果存放在这里')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

devkit_path = "./" + 'RESULT'
set_type = 'test'


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec_sixray(img_id, annopath, imgpath):
    annop = [x for x in os.listdir(annopath) if x.find(img_id) != -1]
    img = [x for x in os.listdir(imgpath) if x.find(img_id) != -1]
    filename = os.path.join(annopath, annop[0])
    imagename1 = os.path.join(imgpath, img[0])
    """ Parse a PASCAL VOC xml file """
    lines = open(filename, mode="r", encoding="utf-8").readlines()
    # strs = line.split(" ")
    # 还需要同时打开图像，读入图像大小
    img = cv2.imread(imagename1)
    # if img is None:
    #     img = cv2.imread(imagename2)
    height, width, channels = img.shape
    objects = []
    for obj in lines:
        temp = obj.split(" ")
        obj_struct = {}
        obj_struct['name'] = "core" if "带电芯充电宝" == temp[1] else "coreless"
        # obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = 0
        obj_struct['difficult'] = 0
        # bbox = obj.find('bndbox')
        xmin = int(temp[2])
        # 只读取V视角的
        if int(xmin) > width:
            continue
        if xmin < 0:
            xmin = 1
        ymin = int(temp[3])
        if ymin < 0:
            ymin = 1
        xmax = int(temp[4])
        if xmax > width:
            xmax = width - 1
        ymax = int(temp[5])
        if ymax > height:
            ymax = height - 1
        obj_struct['pose'] = 'Unspecified'
        obj_struct['truncated'] = 0
        obj_struct['difficult'] = 0
        obj_struct['bbox'] = [float(xmin) - 1,
                              float(ymin) - 1,
                              float(xmax) - 1,
                              float(ymax) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def get_sixray_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def get_result_file():
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, "result.txt")
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind + 1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def write_sixray_results_file(all_boxes, dataset):
    f_result = open(get_result_file(), "wt")
    for cls_ind, cls in enumerate(labelmap):
        f_result.write("================%s===============\n" % cls)
        print('Writing {:s} SIXRAY results file'.format(cls))
        filename = get_sixray_results_file_template(set_type, cls)
        print("Filename: {:s}".format(filename))

        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind + 1][im_ind]

                f_result.write('\nGROUND TRUTH FOR: ' + index + '\n')
                img_id, annotation = dataset.pull_anno(im_ind)
                for box in annotation:
                    f_result.write(str(box[-1]) + ' label: ' + labelmap[box[-1]] + ' ' + ' || '.join(
                        str(b) for b in box[:-1]) + '\n')
                pred_num = 0
                for k in range(len(dets)):
                    if dets[k, -1] >= 0.5:
                        f_result.write('PREDICTIONS: ' + '\n')
                        score = dets[k, -1]
                        label_name = cls
                        coords = (dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3])
                        f_result.write(str(cls_ind) + ' label: ' + label_name + ' ' + ' || '.join(
                            str(c) for c in coords) + ' score: ' +
                                       str(score) + '\n')

                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0], dets[k, 1],
                                   dets[k, 2], dets[k, 3]))


# def do_python_eval(output_dir='output', use_07=True):
def do_python_eval(dataset, annopath, imgpath):
    # cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # if not os.path.isdir(output_dir):
    #     os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        # 每个分类的预测框所在的文件
        filename = get_sixray_results_file_template(set_type, cls)
        rec, prec, ap = sixray_eval(
            filename, annopath, imgpath, "", cls,
            ovthresh=0.5, dataset=dataset)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        # with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        #     pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def sixray_ap(rec, prec):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def sixray_eval(detpath,
                annopath,
                imgpath,
                imagesetfile,
                classname,
                ovthresh=0.5,
                dataset=None
                ):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
    # if not os.path.isdir(cachedir):
    #     os.mkdir(cachedir)
    # cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    # with open(imagesetfile, 'r') as f:
    #     lines = f.readlines()
    lines = dataset.ids
    imagenames = [x.strip() for x in lines]
    # load annots
    recs = {}
    for i, img_id in enumerate(imagenames):
        annop = [x for x in os.listdir(annopath) if x.find(img_id) != -1]
        recs[img_id] = parse_rec_sixray(img_id, annopath, imgpath)
        if i % 100 == 0:
            print('Reading annotation for {:d}/{:d}'.format(
                i + 1, len(imagenames)))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)

        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = sixray_ap(rec, prec)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


# TODO
def test_net(net, dataset, transform, annopath, imgpath):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap) + 1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    # output_dir = get_output_dir('ssd300_120000', set_type)
    # det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    # with open(det_file, 'wb') as f:
    #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, dataset, annopath, imgpath)


def evaluate_detections(box_list, dataset, annopath, imgpath):
    write_sixray_results_file(box_list, dataset)
    do_python_eval(dataset, annopath, imgpath)


def test_(imgpath, annopath):
    # load net
    num_classes = len(labelmap) + 1  # +1 for background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    if not args.cuda:
        net.load_state_dict(torch.load(args.trained_model, map_location='cpu'))
    else:
        net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    # dataset = VOCDetection(args.voc_root, [('2007', set_type)],
    #                        BaseTransform(300, dataset_mean),
    #                        VOCAnnotationTransform())
    dataset_mean = (104, 117, 123)
    # set_type = 'test'
    dataset = SIXrayDetectionEval(imgpath=imgpath, annopath=annopath,
                                  images_set_file=None,
                                  transform=BaseTransform(300, dataset_mean),
                                  target_transform=SIXrayAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(net, dataset, BaseTransform(net.size, dataset_mean), annopath, imgpath)


if __name__ == "__main__":
    test_( r"C:\Work\test\Image_test",r"C:\Work\test\Anno_test")
