from utils.utils import *
from utils.YOLODataLoader import *
import torch.optim as optim
from model.YOLO import YOLO
import argparse
import sys
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO V3')
    parser.add_argument('--batch_size',type=int,default=16,help='size of each batch')
    parser.add_argument('--img_size',type=int,default=416,help='size of input image')   
    parser.add_argument('--momentum',type=float,default=0.9,help='momentum of SGD')
    parser.add_argument('--decay',type=float,default=0.0005,help='decay of SGD')
    parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate of optimizers')
    parser.add_argument('--epochs',type=int,default=30, help='number of epochs')
    parser.add_argument('--n_cpu',type=int,default=4,help='number of cpu threads creating dataloader')
    parser.add_argument('--model_config_path',type=str,default='cfg/handsup.cfg',help='location of model config file')
    parser.add_argument('--class_path', type=str, default='data/handsup.names', help='path to class label file')
    parser.add_argument('--data_config_path',type=str,default='cfg/handsup.data',help='location of data config file')
    parser.add_argument('--checkpoint_path',type=str,default='checkpoint',help='location of checkpoints')
    parser.add_argument('--confidence',type=float,default=0.8,help='object confidence threshold')
    parser.add_argument('--nms_thresh',type=float,default=0.4,help='IOU threshold for non-maxumum suppression')
    parser.add_argument('--use_GPU',type=bool,default=True,help='if use GPU for training')

    parameters = parser.parse_args()
    print(parameters)

    data_config = parseDataConfig(parameters.data_config_path)
    num_classes = int(data_config['classes'])

    dataloader = getDataLoader(parameters, data_config['train'], True)

    model = YOLO(parameters.model_config_path,num_classes)

    CUDA = torch.cuda.is_available() and parameters.use_GPU

    if CUDA:
        model = model.cuda()

    FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), parameters.learning_rate)
    min_loss = float('inf')

    for epoch in range(parameters.epochs):
        for batch_idx, (_, imgs, targets) in enumerate(dataloader):
            imgs = imgs.type(FloatTensor)
            targets = targets.type(FloatTensor)
            optimizer.zero_grad()
            loss = model(imgs, CUDA, targets)
            loss.backward()
            optimizer.step()

            print('Epoch:{}, Batch:{}, x_loss:{:0.4f}, y_loss:{:.4f}, w_loss:{:.4f}, h_loss:{:.4f}, conf:{:.4f},cls:{:.4f}, precision:{:.4f},recall:{:.4f}, total:{:.4f}'\
                .format(epoch, batch_idx,\
                model.losses["x"],model.losses["y"],\
                model.losses["w"],model.losses["h"],\
                model.losses["conf"],model.losses["cls"],\
                model.losses["recall"],model.losses["precision"],\
                loss.item()))

            model.seen += imgs.size(0)

            if loss.item() < min_loss:
                print('Better model found, saving it...')
                for f in Path(parameters.checkpoint_path).glob('*.weights'):
                    f.unlink()
                min_loss = loss.item()
                model.saveModel('{}/{:.4f}.weights'.format(parameters.checkpoint_path,min_loss))
                print('Saved!')
