import torch
import torch.optim as optim
import torch.nn as nn
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from model.Encoder import Encoder
from model.Decoder import Decoder
import os.path as osp
import cv2
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image


def get_arg():
    parser = argparse.ArgumentParser(description="AutoEncoder")
    parser.add_argument("--data-dir", type=str,
                            default="../Warehouse/atari_data/training", help="path to dataset")
    parser.add_argument("--checkpoint-dir", type=str,
                            default="./checkpoints/", help="path to store checkpoint")
    parser.add_argument("--restore", type=bool,
                            default=False, help="load pretrained model or not")
    parser.add_argument("--batch-size", type=int,
                            default=125, help="training batch size")
    parser.add_argument("--model-dir", type=str,
                            default="./checkpoints/", help="path in which pretrained weights are stored")
    parser.add_argument("--training", type=bool,
                            default=True, help="to perform training(true) or testing(false)")
    parser.add_argument("--experiment_id", type=int,
                            default=0, help="identification number of this experiment")
    parser.add_argument("--train-size", type=int,
                            default=20000, help="size of training data per load")
    parser.add_argument("--validation-size", type=int,
                            default=10000, help="size of validation data per load")
    parser.add_argument("--epoch", type=int,
                            default=5, help="number of epoch")
    parser.add_argument("--learning-rate", type=float,
                            default=0.001, help="learning rate of the optimizer")
    parser.add_argument("--gpu", type=int,
                            default=0, help="id of the gpu to run the program on")
    parser.add_argument("--image-dir", type=str, 
                            default="./image", help="path to store generated images")
    return parser.parse_args()
    
def process_frame(frame, shape=(84, 84)):
    frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[34:34+160, :160]  # crop image
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))
    cv2.imwrite('gray.png', frame)
    #print(frame.shape)
    return frame

def get_data(index):
    train_img = [] # (4-d tensor) shape : size, w, h, 3
    valid_img = []
    batch_img = [] #(4-d tensor) shape: 4, w, h
    directory = args.data_dir
    train_st, train_ed = index, index + args.train_size
    valid_st, valid_ed = train_ed + 1, train_ed + args.validation_size
    for i in range(train_st, train_ed+1):
        if i % 1000 == 0:
            print("loading {:d}-th img".format(i))
        path = directory + '/{:06d}.png'.format(i)
        img = cv2.imread(path)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #img = cv2.resize(img, (104, 80), interpolation=cv2.INTER_NEAREST)
        img = process_frame(img)
        img = img / 255
        train_img.append(np.expand_dims(np.array(img, dtype=np.float32), axis=0))
        # batch_img.append(img)
        # if len(batch_img) == 4:
        #     train_img.append(np.array(batch_img, dtype=np.float32))
        #     batch_img = []
    train_img = np.array(train_img, dtype=np.float32)
    train_img = np.squeeze(train_img, axis=4)
    print(train_img.shape)
    for i in range(valid_st, valid_ed+1):
        if i % 1000 == 0:
            print("loading {:d}-th img".format(i))
        path = directory + '/{:06d}.png'.format(i)
        img = cv2.imread(path)
        img = process_frame(img)
        img = img / 255
        # batch_img.append(img)
        # if len(batch_img) == 4:
        #     valid_img.append(np.array(batch_img, dtype=np.float32))
        #     batch_img = []
        valid_img.append(np.expand_dims(np.array(img, dtype=np.float32), axis=0))
    valid_img = np.array(valid_img, dtype=np.float32)
    valid_img = np.squeeze(valid_img, axis=4)
    print(valid_img.shape)  
    return train_img, valid_img

#parse command line arguments
args = get_arg()

def main():
    
    #create tensorboard summary writer
    writer = SummaryWriter(args.experiment_id)
    #[TODO] may need to resize input image
    cudnn.enabled = True
    #create model: Encoder
    model_encoder = Encoder()
    model_encoder.train()
    model_encoder.cuda(args.gpu)
    optimizer_encoder = optim.Adam(model_encoder.parameters(), lr=args.learning_rate, betas=(0.95, 0.99))
    optimizer_encoder.zero_grad()

    #create model: Decoder
    model_decoder = Decoder()
    model_decoder.train()
    model_decoder.cuda(args.gpu)
    optimizer_decoder = optim.Adam(model_decoder.parameters(), lr=args.learning_rate, betas=(0.95, 0.99))
    optimizer_decoder.zero_grad()
    
    l2loss = nn.MSELoss()
    
    #load data
    for i in range(1, 360002, 30000):
        train_data, valid_data = get_data(i)
        for e in range(1, args.epoch + 1):
            train_loss_value = 0
            validation_loss_value = 0
            for j in range(0, int(args.train_size/4), args.batch_size):
                optimizer_decoder.zero_grad()
                optimizer_decoder.zero_grad()
                image = Variable(torch.tensor(train_data[j: j + args.batch_size, :, :])).cuda(args.gpu)
                latent = model_encoder(image)
                img_recon = model_decoder(latent)
                img_recon = F.interpolate(img_recon, size=image.shape[2:], mode='bilinear', align_corners=True) 
                loss = l2loss(img_recon, image)
                train_loss_value += loss.data.cpu().numpy() / args.batch_size
                loss.backward()
                optimizer_decoder.step()
                optimizer_encoder.step()
            print("data load: {:8d}".format(i))
            print("epoch: {:8d}".format(e))
            print("train_loss: {:08.6f}".format(train_loss_value / (args.train_size / args.batch_size)))
            for j in range(0,int(args.validation_size/4), args.batch_size):
                model_encoder.eval()
                model_decoder.eval() 
                image = Variable(torch.tensor(valid_data[j: j + args.batch_size, :, :])).cuda(args.gpu)
                latent = model_encoder(image)
                img_recon = model_decoder(latent)
                img_1 = img_recon[0][0]
                img = image[0][0]
                img_recon = F.interpolate(img_recon, size=image.shape[2:], mode='bilinear', align_corners=True) 
                save_image(img_1, args.image_dir + '/fake' + str(i) + "_" + str(j) + ".png")
                save_image(img, args.image_dir + '/real' + str(i) + "_" + str(j) + ".png")
                image = Variable(torch.tensor(train_data[j: j + args.batch_size, :, :, :])).cuda(args.gpu)
                loss = l2loss(img_recon, image)
                validation_loss_value += loss.data.cpu().numpy() / args.batch_size
            model_encoder.train()
            model_decoder.train()
            print("train_loss: {:08.6f}".format(validation_loss_value / (args.validation_size / args.batch_size)))
        torch.save({'encoder_state_dict': model_encoder.state_dict()}, osp.join(args.checkpoint_dir, 'AE_encoder.pth'))
        torch.save({'decoder_state_dict': model_decoder.state_dict()}, osp.join(args.checkpoint_dir, 'AE_decoder.pth'))

if __name__ == "__main__":
    main()
                

                


        

    












