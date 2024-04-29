import torch
from SKNET import sknet29
from Resnet import resnet18, resnet34
from Resnext import resnext29, resnext50
import utils
from train_test import train
import argparse
from datetime import datetime 

'''
cmd :
nohup python -u SKNET-PyTorch/main.py -epochs 50 -model 5 -skconv -lr 0.0005 -batchsize 150 &

scp :
scp -P 21490 root@ssh5.vast.ai:SKNET.pt .\Desktop\
'''

'''
Define parser to get groups as cmd input
'''
parser = argparse.ArgumentParser()
parser.add_argument("-G", default = 1, type=int, help="number of conv groups in model")
parser.add_argument("-batchsize", default = 256, type=int, help="batch size")
parser.add_argument("-epochs", default = 30, type=int, help="epochs")
parser.add_argument("-model", default = 1, type=int, help="1 -> SKNET, 2-> ResNet18, 3-> ResNet34, 4-> ResNeXt29, 5-> ResNeXt50")
parser.add_argument("-skconv", action="store_true")
parser.add_argument("-use1x1", action="store_true")
parser.add_argument("-cyclic", action="store_true")
parser.add_argument("-M", default = 2, type=int, help="no. of branches in skconv")
parser.add_argument("-lr", default = 1e-4, type=float, help="lr")
args = parser.parse_args()


'''
Downloading & importing dataset
'''
print('Process Started at %s'%(str(datetime.now())))
utils.download_data()
train_loader, val_loader = utils.get_dataloaders(batch_size = args.batchsize)
print('Data downloaded and loaded sucessfully')

'''
Define SKNet Model
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(device)

num_classes = 200
if int(args.model) == 1:
    print("SKNET, use1x1 = %s"%(args.use1x1))
    net = sknet29(200, True)
    # net = SKNet(200, [2,2,2,2], [1,2,2,2], G = args.G , use_1x1 = args.use1x1, M = args.M)
elif int(args.model) == 2:
    print("ResNet18, skconv = %s, use1x1 = %s"%(args.skconv, args.use1x1))
    net = resnet18(200, args.skconv, args.use1x1)
elif int(args.model) == 3:
    print("ResNet34, skconv = %s, use1x1 = %s"%(args.skconv, args.use1x1))
    net = resnet34(200, args.skconv, args.use1x1)
elif int(args.model) == 4:
    print("ResNeXt29, skconv = %s, use1x1 = %s"%(args.skconv, args.use1x1))
    net = resnext29(200, args.skconv, args.use1x1, groups = 32, width_per_group=4)
elif int(args.model) == 5:
    print("ResNeXt50, skconv = %s, use1x1 = %s"%(args.skconv, args.use1x1))
    net = resnext50(200, args.skconv, args.use1x1, groups = 32, width_per_group=4)
else:
    print("Wrong model input, check help")
    parser.print_help()
    
net.to(device)
print("# of Parameters : ",sum([p.numel() for p in net.parameters()]))
print("Model loaded sucessfully")

'''
Start Training
'''

net = train(net, 
            device, 
            train_loader, 
            val_loader, 
            lr = float(args.lr), 
            cyclic = args.cyclic, 
            epochs = int(args.epochs))
# test(net, device, test_loader)