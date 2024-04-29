import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from loss import YoloLoss
from model.backbones import vgg16
from model import yolov1
from datasets import geo_shape
import test

#python eval_testset.py -i "D:\\datasets\\geo_shapes\\train" -j "D:\\datasets\\geo_shapes\\labels.json" -a "train_val_test_idxs" -m "best.pth.tar" 

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--img_root",type=str,help="Image directory")
    parser.add_argument("-j","--json_path",type=str,help="path to labels json")
    parser.add_argument("-a","--anno_dir",type=str,help="Directory contain npy idxs")
    parser.add_argument("-m","--model_path",type=str,help="Path to saved model")
    return parser.parse_args()

if __name__ == '__main__':
    args = arguments()
    img_root = args.img_root
    json_path = args.json_path
    idx_dir = args.anno_dir
    model_path = args.model_path
    device = "cpu"

    #Get test dataset
    test_dataset = geo_shape.GeoShape(img_root,idx_dir,json_path,mode='test')
    test_loader = DataLoader(test_dataset,batch_size=1)

    #get backbone and model
    backbone = vgg16.VGG16(in_c=1)
    model = yolov1.YOLOv1(backbone)

    #load checkpoint
    checkpoint = torch.load(model_path,map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    test_loss,test_iou = test.run(test_loader,model,device)
    print("Test Loss : {} , Test IOU : {}".format(test_loss,test_iou))

