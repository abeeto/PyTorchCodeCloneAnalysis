import os, time
import cProfile
import openface
import torch
import torchfile
from OpenFacePytorch.loadOpenFace import prepareOpenFace

imgDim = 96
NEW = True
import cv2, numpy

def ReadImage(pathname):
    img = cv2.imread(pathname)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)
    img = numpy.transpose(img, (2, 0, 1))
    img = img.astype(numpy.float32) / 255.0
    I_ = torch.from_numpy(img).unsqueeze(0)

    return I_

# Openface: forward image to torch subprocess
def traditional_predict(img_path):
    rep = net.forwardPath(img_path)
    return rep


# Proposed: Directly use pytorch
def new_predict(img_path):
    rep = torch_net(ReadImage(img_path))
    return rep[0].tolist()[0]


if __name__ == '__main__':

    global imgDim
    # define paths
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models/nn4.small2.v1.t7')
    dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')

    result_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/old' if not NEW else 'results/new' )
    measurement_path = os.path.join(result_path, 'measurements.csv')
    rep_path = os.path.join(result_path, 'rep.csv')

    net = openface.TorchNeuralNet(model_path, imgDim=imgDim, cuda=False)
    torch_net = prepareOpenFace(useCuda=False).eval()

    if not os.path.exists('results'):
        os.mkdir('results')

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    paths = [measurement_path, rep_path]
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


    for idx, img in enumerate(os.listdir(dataset_path)):
        profiler = cProfile.Profile()
        profiler.enable()
        tick = time.time()
        
        if NEW:
            rep = new_predict(os.path.join(dataset_path, img))
        else:
            rep = traditional_predict(os.path.join(dataset_path, img))

        with open(measurement_path, 'a') as f:
            delta = str(time.time() - tick)
            print("Elapse time: " + delta + '(s)')
            f.write(delta + '\n')

        profiler.disable()
        profiler.create_stats()
        profiler.dump_stats(os.path.join(result_path, '{}.pyprof'.format(idx)))

        with open(rep_path, 'a') as f:
            f.write(str(rep) + '\n')
