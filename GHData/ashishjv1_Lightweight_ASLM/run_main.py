from __future__ import print_function, division
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import time
import numpy as np
import models as models
import models_cpu as models_cpu
import torch.backends.cudnn as cudnn
import argparse

parser = argparse.ArgumentParser(description="Create Model From Ranks")
parser.add_argument('--data_path',
                    default='',
                    type=str, required=True, help='Path to datset')
parser.add_argument('--train_label',
                    default='', type=str,
                    required=True, help='Train Labels')
parser.add_argument('--test_label',
                    default='', type=str,
                    required=True, help='Test Labels')
parser.add_argument('--eps', default='1e-12', type=float, required=False, help='EPS Value')
parser.add_argument('--model', default='full', type=str, required=True, help='Model to Test Full(F)/Compressed(c)')
parser.add_argument('--attr_num', default='26', type=int, required=True, help='(35)PETA or (26)PA-100K')
parser.add_argument('--experiment', default='PA-100K', type=str, required=True, help='PETA(PETA/peta) or PA-100K(PA)')
parser.add_argument('--model_path', default='',
                    type=str, required=True, help='path to saved Compressed model')
parser.add_argument('--checkpoint_save', default='/checkpoints/', type=str,
                    required=True, help='checkpoint save path')
parser.add_argument('--checkpoint_load', default='', type=str, required=False, help='checkpoint load path')
parser.add_argument('--epoch', default='15', type=int, required=False, help='Number of Epochs to train the Network')
parser.add_argument('--device', default='cuda', type=str, required=True, help='Device to Use CUDA or CPU')
parser.add_argument('--loss_accu', default='/tmp/', type=str, required=False,
                    help='save losses and accuracy to a directory')
args = parser.parse_args()

data_path = args.data_path
train_list_path = args.train_label
val_list_path = args.test_label
EPS = args.eps
device = args.device


####DATA LOADER CLASS###

def default_loader(path):
    return Image.open(path).convert('RGB')


class MultiLabelDataset(data.Dataset):
    def __init__(self, root, label, transform=None, loader=default_loader):
        images = []
        labels = open(label).readlines()
        for line in labels:
            items = line.split()
            img_name = items.pop(0)
            if os.path.isfile(os.path.join(root, img_name)):
                cur_label = tuple([int(v) for v in items])
                images.append((img_name, cur_label))
            else:
                print(os.path.join(root, img_name) + 'Not Found.')
        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name, label = self.images[index]
        img = self.loader(os.path.join(self.root, img_name))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.images)


###DATA PROCESSING

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.Resize(size=(256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
transform_test = transforms.Compose([
    transforms.Resize(size=(256, 128)),
    transforms.ToTensor(),
    normalize
])

train_dataset = MultiLabelDataset(root=data_path, label=train_list_path, transform=transform_train)
val_dataset = MultiLabelDataset(root=data_path, label=val_list_path, transform=transform_test)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    batch_size = target.size(0)
    attr_num = target.size(1)

    output = torch.sigmoid(output).cpu().numpy()
    output = np.where(output > 0.5, 1, 0)
    pred = torch.from_numpy(output).long()
    target = target.cpu().long()
    correct = pred.eq(target)
    correct = correct.numpy()
    res = []
    for k in range(args.attr_num):
        res.append(1.0 * sum(correct[:, k]) / batch_size)
    return sum(res) / args.attr_num


##TRAIN Function
def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    print_freq = 500
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    end = time.time()
    for i, _ in enumerate(train_loader):
        input, target = _
        target = target.to(device)
        input = input.to(device)
        output = model(input)
        bs = target.size(0)

        if type(output) == type(()) or type(output) == type([]):
            loss_list = []
            # deep supervision
            for k in range(len(output)):
                out = output[k]
                loss_list.append(criterion.forward(torch.sigmoid(out), target))
            loss = sum(loss_list)
            # maximum voting
            output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])
        else:
            loss = criterion.forward(torch.sigmoid(output), target)

        # measure accuracy and record loss
        accu = accuracy(output.data, target)
        losses.update(loss.data, bs)
        top1.update(accu, bs)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accu {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))
    return losses.avg


##VALIDATE FUNCTION
def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    print_freq = 500
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()

    end = time.time()
    for i, _ in enumerate(val_loader):
        input, target = _
        target - target.to(device)
        input = input.to(device)
        output = model(input)

        bs = target.size(0)

        if type(output) == type(()) or type(output) == type([]):
            loss_list = []
            # deep supervision
            for k in range(len(output)):
                out = output[k]
                loss_list.append(criterion.forward(torch.sigmoid(out), target))
            loss = sum(loss_list)
            # maximum voting
            output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])
        else:
            loss = criterion.forward(torch.sigmoid(output), target)

        # measure accuracy and record loss
        accu = accuracy(output.data, target)
        losses.update(loss.data, bs)
        top1.update(accu, bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accu {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Accu {top1.avg:.3f}'.format(top1=top1))
    return top1.avg, losses.avg


##TEST Function
def test(val_loader, model, attr_num, description, epoch):
    model.eval()
    pos_cnt = []
    pos_tol = []
    neg_cnt = []
    neg_tol = []

    accu = 0.0
    prec = 0.0
    recall = 0.0
    tol = 0

    for it in range(attr_num):
        pos_cnt.append(0)
        pos_tol.append(0)
        neg_cnt.append(0)
        neg_tol.append(0)

    for i, _ in enumerate(val_loader):
        input, target = _
        target - target.to(device)
        input = input.to(device)
        output = model(input)
        bs = target.size(0)

        # maximum voting
        if type(output) == type(()) or type(output) == type([]):
            output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])

        batch_size = target.size(0)
        tol = tol + batch_size
        output = torch.sigmoid(output.data).cpu().numpy()
        output = np.where(output > 0.5, 1, 0)
        target = target.cpu().numpy()

        for it in range(attr_num):
            for jt in range(batch_size):
                if target[jt][it] == 1:
                    pos_tol[it] = pos_tol[it] + 1
                    if output[jt][it] == 1:
                        pos_cnt[it] = pos_cnt[it] + 1
                if target[jt][it] == 0:
                    neg_tol[it] = neg_tol[it] + 1
                    if output[jt][it] == 0:
                        neg_cnt[it] = neg_cnt[it] + 1

        if attr_num == 1:
            continue
        for jt in range(batch_size):
            tp = 0
            fn = 0
            fp = 0
            for it in range(attr_num):
                if output[jt][it] == 1 and target[jt][it] == 1:
                    tp = tp + 1
                elif output[jt][it] == 0 and target[jt][it] == 1:
                    fn = fn + 1
                elif output[jt][it] == 1 and target[jt][it] == 0:
                    fp = fp + 1
            if tp + fn + fp != 0:
                accu = accu + 1.0 * tp / (tp + fn + fp)
            if tp + fp != 0:
                prec = prec + 1.0 * tp / (tp + fp)
            if tp + fn != 0:
                recall = recall + 1.0 * tp / (tp + fn)

    print('=' * 100)
    print('\t     Attr              \tp_true/n_true\tp_tol/n_tol\tp_pred/n_pred\tcur_mA')
    mA = 0.0
    for it in range(attr_num):
        cur_mA = ((1.0 * pos_cnt[it] / pos_tol[it]) + (1.0 * neg_cnt[it] / neg_tol[it])) / 2.0
        mA = mA + cur_mA
        print('\t#{:2}: {:18}\t{:4}\{:4}\t{:4}\{:4}\t{:4}\{:4}\t{:.5f}'.format(it, description[it], pos_cnt[it],
                                                                               neg_cnt[it], pos_tol[it], neg_tol[it], (
                                                                                       pos_cnt[it] + neg_tol[it] -
                                                                                       neg_cnt[it]), (
                                                                                       neg_cnt[it] + pos_tol[it] -
                                                                                       pos_cnt[it]), cur_mA))
    mA = mA / attr_num
    print('\t' + 'mA:        ' + str(mA))

    if attr_num != 1:
        accu = accu / tol
        prec = prec / tol
        recall = recall / tol
        f1 = 2.0 * prec * recall / (prec + recall)
        print('\t' + 'Accuracy:  ' + str(accu))
        print('\t' + 'Precision: ' + str(prec))
        print('\t' + 'Recall:    ' + str(recall))
        print('\t' + 'F1_Score:  ' + str(f1))
    print('=' * 100)


## Adjust Learning Rate Function
def adjust_learning_rate(optimizer, epoch, decay_epoch):
    lr = 0.0001
    for epc in decay_epoch:
        if epoch >= epc:
            lr = lr * 0.1
        else:
            break
    print()
    print('Learning Rate:', lr)
    print()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Weighted_BCELoss(object):
    """
        Weighted_BCELoss was taken from "Multi-attribute learning for pedestrian attribute recognition in surveillance scenarios".
    """

    def __init__(self, experiment):
        super(Weighted_BCELoss, self).__init__()
        self.weights = None
        if experiment == 'pa100k' or experiment == 'PA-100K' or experiment == 'PA':
            self.weights = torch.Tensor([0.460444444444,
                                         0.0134555555556,
                                         0.924377777778,
                                         0.0621666666667,
                                         0.352666666667,
                                         0.294622222222,
                                         0.352711111111,
                                         0.0435444444444,
                                         0.179977777778,
                                         0.185,
                                         0.192733333333,
                                         0.1601,
                                         0.00952222222222,
                                         0.5834,
                                         0.4166,
                                         0.0494777777778,
                                         0.151044444444,
                                         0.107755555556,
                                         0.0419111111111,
                                         0.00472222222222,
                                         0.0168888888889,
                                         0.0324111111111,
                                         0.711711111111,
                                         0.173444444444,
                                         0.114844444444,
                                         0.006]).to(device)

        elif experiment == 'peta' or experiment == 'PETA' or experiment == 'P':
            self.weights = torch.Tensor([0.5016,
                                         0.3275,
                                         0.1023,
                                         0.0597,
                                         0.1986,
                                         0.2011,
                                         0.8643,
                                         0.8559,
                                         0.1342,
                                         0.1297,
                                         0.1014,
                                         0.0685,
                                         0.314,
                                         0.2932,
                                         0.04,
                                         0.2346,
                                         0.5473,
                                         0.2974,
                                         0.0849,
                                         0.7523,
                                         0.2717,
                                         0.0282,
                                         0.0749,
                                         0.0191,
                                         0.3633,
                                         0.0359,
                                         0.1425,
                                         0.0454,
                                         0.2201,
                                         0.0178,
                                         0.0285,
                                         0.5125,
                                         0.0838,
                                         0.4605,
                                         0.0124]).to(device)
        # self.weights = None

    def forward(self, output, target):
        if self.weights is not None:
            cur_weights = torch.exp(target + (1 - target * 2) * self.weights)
            loss = cur_weights * (target * torch.log(output + EPS)) + ((1 - target) * torch.log(1 - output + EPS))
        else:
            loss = target * torch.log(output + EPS) + (1 - target) * torch.log(1 - output + EPS)
        return torch.neg(torch.mean(loss))


def save_checkpoint(state, epoch, prefix, filename='.pth.tar'):
    """Saves checkpoint to disk"""
    path = args.checkpoint_save
    experiment = args.experiment
    approach = 'inception'
    directory = path + experiment + '/' + approach + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    if prefix == '':
        filename = directory + str(epoch) + filename
    else:
        filename = directory + prefix + '_' + str(epoch) + filename
    torch.save(state, filename)


def main():
    t_losses = []
    v_losses = []
    accuracy = []
    prefix = "Best_Accuracy"
    attr_num = args.attr_num
    lr = 0.0001
    weight_decay = 0.0005
    start_epoch = 0
    epochs = args.epoch
    decay_epoch = (int((start_epoch + epochs) / 2), epochs - 5)
    batch_size = 32
    # resume_check = args.checkpoint_load
    if args.experiment == "PA-100K" or args.experiment == "PA" or args.experiment == "pa-100k":
        description = ['Female',
                       'AgeOver60',
                       'Age18-60',
                       'AgeLess18',
                       'Front',
                       'Side',
                       'Back',
                       'Hat',
                       'Glasses',
                       'HandBag',
                       'ShoulderBag',
                       'Backpack',
                       'HoldObjectsInFront',
                       'ShortSleeve',
                       'LongSleeve',
                       'UpperStride',
                       'UpperLogo',
                       'UpperPlaid',
                       'UpperSplice',
                       'LowerStripe',
                       'LowerPattern',
                       'LongCoat',
                       'Trousers',
                       'Shorts',
                       'Skirt&Dress',
                       'boots']
    elif args.experiment == "PETA" or args.experiment == "peta" or args.experiment == 'P':
        description = ['Age16-30',
                       'Age31-45',
                       'Age46-60',
                       'AgeAbove61',
                       'Backpack',
                       'CarryingOther',
                       'Casual lower',
                       'Casual upper',
                       'Formal lower',
                       'Formal upper',
                       'Hat',
                       'Jacket',
                       'Jeans',
                       'Leather Shoes',
                       'Logo',
                       'Long hair',
                       'Male',
                       'Messenger Bag',
                       'Muffler',
                       'No accessory',
                       'No carrying',
                       'Plaid',
                       'PlasticBags',
                       'Sandals',
                       'Shoes',
                       'Shorts',
                       'Short Sleeve',
                       'Skirt',
                       'Sneaker',
                       'Stripes',
                       'Sunglasses',
                       'Trousers',
                       'Tshirt',
                       'UpperOther',
                       'V-Neck']

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    if args.model == 'full' or args.model == 'FULL' or args.model == 'f' or args.model == 'F':
        if args.device == "cuda":
            model = models.__dict__["inception_iccv"](pretrained=True, num_classes=attr_num)
            model = torch.nn.DataParallel(model).to(device)
            print('Number of model parameters: {}'.format(
                sum([p.numel() for p in model.parameters()])))
            print('')
        else:
            model = models_cpu.__dict__["inception_iccv"](pretrained=True, num_classes=attr_num)
            print('Number of model parameters: {}'.format(
                sum([p.numel() for p in model.parameters()])))
            print('')

    else:

        PATH = args.model_path
        model = torch.load(PATH)

        if args.device == "cuda":
            model.eval()
            model = torch.nn.DataParallel(model).to(device)
            print('Number of model parameters: {}'.format(
                sum([p.numel() for p in model.parameters()])))
            print('')
        else:
            model.eval()
    # optionally resume from a checkpoint

    #     if resume_check:
    #         if os.path.isfile(resume_check):
    #             print("=> loading checkpoint '{}'".format(resume_check))
    #             checkpoint = torch.load(resume_check)
    #             start_epoch = checkpoint['epoch']
    #             best_accu = checkpoint['best_accu']
    #             model.load_state_dict(checkpoint['state_dict'])
    #             print("=> loaded checkpoint '{}' (epoch {})"
    #                   .format(resume_check, checkpoint['epoch']))
    #         else:
    #             print("=> no checkpoint found at '{}'".format(resume_check))

    cudnn.benchmark = False
    cudnn.deterministic = True
    criterion = Weighted_BCELoss(args.experiment)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 betas=(0.9, 0.999),
                                 weight_decay=weight_decay)

    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch, decay_epoch)
        # train for one epoch
        train_losses = train(train_loader, model, criterion, optimizer, epoch)
        t_losses.append(train_losses)
        torch.cuda.empty_cache()
        # evaluate on validation set
        accu, validation_losses = validate(val_loader, model, criterion, epoch)
        accu = float(accu)
        accuracy.append(accu)
        v_losses.append(validation_losses)
        test(val_loader, model, attr_num, description, epoch)

        # remember best Accu and save checkpoint
        # is_best = accu > best_accu
        # best_accu = max(accu, best_accu)

        if epoch in decay_epoch:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_accu': accu,    ##'best_accu': best_accu,
            }, epoch + 1, prefix)
    return t_losses, v_losses, accuracy


if __name__ == '__main__':
    main()
    train_loss, validation_loss, accuracy = main()
    save_directory = args.loss_accu + '/'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    save_as_dict = {"training_losses": train_loss, "validation_losses": validation_loss, "accuracy": accuracy}
    np.save(save_directory + "Losses_and_Accuracy", save_as_dict)
