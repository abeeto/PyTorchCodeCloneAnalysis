import time
import torch 
from torch.autograd import Variable
import cv2 
from scripts.util import *
from scripts.darknet import Darknet
from scripts.preprocess import prep_image
import argparse
import random

class Detection:
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    classes = load_classes('data/coco.names')
    num_classes = 80
    bbox_attrs = 5 + num_classes
    args = None
    confidence = None
    nms_thesh = None
    CUDA = False
    model = None
    inp_dimensions = None
    colors = list()

    def color_generator(self):
        ''' Generate a color pallete for different objects.
        '''

        for i in range(0, 80):
            temp = list()
            b = random.randint(0, 255)
            g = random.randint(0, 255)
            r = random.randint(0, 255)
            temp.append(b)
            temp.append(g)
            temp.append(r)
            self.colors.append(temp)

    def plot_results(self, output, img):
        ''' 
        Draw the bounding box and results on the frame.
        '''

        for x in output:

            if float(x[5]) <= 0 or float(x[5]) > 1:
                continue

            cls = int(x[-1])
            if int(cls)<0 or int(cls)>80:
                continue

            c1 = tuple(x[1:3].int())
            c2 = tuple(x[3:5].int())

            x1, y1 = c1[0].item(), c1[1].item()
            x2, y2 = c2[0].item(), c2[1].item()
                
            label = "{0}".format(self.classes[cls])
            score = str("{0:.3f}".format(float(x[5])))
            color = self.colors[cls]
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y1), (x1 + (len(label) + len(score)) * 10, 
                            y1 - 10) , color, -1, cv2.LINE_AA)
            cv2.putText(img, label + ':' + score, (x1, y1), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        
        return img

    def argsparser(self):
        '''
        Argument parser for command line arguments.
        '''

        parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
        parser.add_argument('--confidence', dest='confidence', help='Object Confidence to filter predictions', default=0.25, type=float)
        parser.add_argument('--nms_thresh', dest='nms_thresh', help='NMS Threshhold', default=0.4, type=float)
        parser.add_argument('--reso', dest='reso', help=
                            'Input resolution of the network. Increase to increase accuracy. Decrease to increase speed',
                            default=416, type=int)
        parser.add_argument('--source', dest='source', default=0, help='Input video source', type=str)
        parser.add_argument('--skip', dest='skip', default=1, help='Frame skip to increase speed', type=int)
        return parser.parse_args()
    
    def run(self):
        '''
        Method to run the detection.
        '''

        cap = cv2.VideoCapture(self.args.source)
        assert cap.isOpened(), 'Cannot capture source'

        frames = 0
        start = time.time()  

        while cap.isOpened():
        
            ret, frame = cap.read()
            if ret:
                if not frames % self.args.skip == 0:
                    frames += 1
                    continue
                
                img, orig_im, dim = prep_image(frame, self.inp_dimensions)
                im_dim = torch.FloatTensor(dim).repeat(1, 2)                        
            
                if self.CUDA:
                    im_dim = im_dim.cuda()
                    img = img.cuda()
            
                output = self.model(Variable(img), self.CUDA)
                output = write_results(output, self.confidence, self.num_classes, nms = True, nms_conf = self.nms_thesh)

                if not type(output) == int:

                    output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(self.inp_dimensions))/self.inp_dimensions

                    im_dim = im_dim.repeat(output.size(0), 1)
                    output[:,[1,3]] *= frame.shape[1]
                    output[:,[2,4]] *= frame.shape[0]
            
                    orig_im = self.plot_results(output, orig_im)
            
                cv2.imshow(self.windowName, orig_im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)), end='\r')
    
    def __init__(self):
        '''
        Intitialize method to run on class object creation.
        '''
        
        self.args = self.argsparser()

        self.CUDA = torch.cuda.is_available()
        if self.CUDA:
            print('Device Used: ', torch.cuda.get_device_name(0))
            print('Capability: ', torch.cuda.get_device_capability(0))

        self.confidence = float(self.args.confidence)
        self.nms_thesh = float(self.args.nms_thresh)

        self.model = Darknet(self.cfgfile)
        self.model.load_weights(self.weightsfile)
        self.model.net_info["height"] = self.args.reso

        self.inp_dimensions = int(self.model.net_info["height"])

        assert self.inp_dimensions % 32 == 0, 'Input not a multiple of 32'
        assert self.inp_dimensions > 32, 'Input must be larger than 32'


        if self.CUDA:
            self.model.cuda()

        self.model.eval()

        self.color_generator()
        self.windowName = "Object Detection"
        cv2.namedWindow(self.windowName, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

if __name__ == '__main__':

    detection = Detection()
    detection.run()
    