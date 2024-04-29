import torch
from SKNET import sknet29
from Resnet import resnet18, resnet34
from Resnext import resnext29, resnext50
import utils
import argparse
from datetime import datetime 
import matplotlib.pyplot as plt
import torch.nn.functional as F


def topN(pred, y, n):
    k_vals = torch.topk(pred, n).indices
    correct = 0
    for i in range(y.shape[0]):
        if y[i].item() in k_vals[i]:
            correct += 1
    return correct

def accuracy(output, target, k):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

    res = []
    # print(correct)
    correct_k = correct[:k].reshape(-1).float().sum(0)
    # print(correct_k.item())
    return correct_k.item()

def val(net, device, test_loader, k):
    correct = 0
    total = 0
    topN_correct = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            outputs = F.softmax(outputs, dim = 1)
            total += labels.size(0)
            topN_correct += accuracy(outputs, labels, k)
    
    topN_acc = 100* topN_correct / total
    
    return topN_acc

def topK_acc(net, device, val_loader, name = "topk"):
    net.eval()
    topN_accuracy = []

    for i in range(1, 11):
        acc = val(net, device, val_loader, i)
        print("Acc : ",acc)
        topN_accuracy.append(acc)
    torch.cuda.empty_cache()
    print(topN_accuracy)
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('top-k accuracy')
    plt.plot(topN_accuracy)
    plt.savefig( name + ".png", format='png', dpi = 600)
    # plt.show()


if __name__ == '__main__':
    
    '''
    python -u Test.py -path "D:\\Mtech\\Sem 4\\Results\\ResNet-50-32x4d_SKConv\\SKNET.pt" -model 5 -skconv
    '''

    '''
    Define parser to get groups as cmd input
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", default = "D:\\Mtech\\Sem 4\\Results\\SKNet-29\\SKNET.pt", type=str, help="")
    parser.add_argument("-batchsize", default = 10, type=int, help="batch size")
    parser.add_argument("-model", default = 1, type=int, help="1 -> SKNET, 2-> ResNet18, 3-> ResNet34, 4-> ResNeXt29, 5-> ResNeXt50")
    parser.add_argument("-skconv", action="store_true")

    args = parser.parse_args()
    
    topN_accuracy = []
    
    print('Process Started at %s'%(str(datetime.now())))
    utils.download_data()
    train_loader, val_loader = utils.get_dataloaders(batch_size = args.batchsize)
    print('Data downloaded and loaded sucessfully')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    name = "topk"
    
    num_classes = 200
    if int(args.model) == 1:
        print("SKNET")
        net = sknet29(200, True)
        name = "SKNET"
        # net = SKNet(200, [2,2,2,2], [1,2,2,2], G = args.G , use_1x1 = args.use1x1, M = args.M)
    elif int(args.model) == 2:
        print("ResNet18, skconv = %s"%(args.skconv))
        net = resnet18(200, args.skconv)
    elif int(args.model) == 3:
        print("ResNet34, skconv = %s"%(args.skconv))
        net = resnet34(200, args.skconv)
    elif int(args.model) == 4:
        print("ResNeXt29, skconv = %s"%(args.skconv))
        net = resnext29(200, args.skconv, groups = 32, width_per_group=4)
        name = "RESNEXT29"
    elif int(args.model) == 5:
        print("ResNeXt50, skconv = %s"%(args.skconv))
        net = resnext50(200, args.skconv, groups = 32, width_per_group=4)
        name = "RESNEXT50"
    else:
        print("Wrong model input, check help")
        parser.print_help()

    
    if args.skconv :
        name += "_SKCONV"
    # model = sknet29(200, True)
    net.load_state_dict(torch.load(args.path))
    net.to(device)
    
    topK_acc(net, device, val_loader, name)
    torch.cuda.empty_cache()