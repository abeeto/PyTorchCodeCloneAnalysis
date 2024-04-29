from __future__ import division

from PIL import Image
from DarkNet import Darknet
from utils import *


def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=False):
    model.eval()

    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    else:
        print('unknown image type')
        exit(-1)

    if use_cuda:
        img = img.cuda()
    img = Variable(img)

    list_boxes = model(img)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    boxes = list_boxes[0][0] + list_boxes[1][0] + list_boxes[2][0]

    boxes = nms(boxes, nms_thresh)

    return boxes


def detect(args):
    m = Darknet(args.cfgfile)

    m.print_network()
    m.load_weights(args.weightsfile)
    print('Loading weights from %s... Done!' % (args.weightsfile))

    use_cuda = False
    if not args.gpu == -1 and torch.cuda.is_available():
        use_cuda = True
        torch.cuda.set_device(args.gpu)

    if use_cuda:
        m.cuda()

    class_names = load_class_names(args.name_file)

    img = Image.open(args.images).convert('RGB')
    sized = img.resize((m.width, m.height))

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, args.confidence, args.nms_thresh, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (args.images, finish - start))

    plot_boxes(img, boxes, args.det, class_names)


if __name__ == '__main__':
    args = detect_arg_parse()
    detect(args)
