import argparse
import datetime
import os
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from datasets import get_train_datasets
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet50_middle
from models.densenet import densenet121
from models.osnet import osnet_x1_0
from models.units import build_optimizer, get_scheduler
from losses.losses import CrossEntropyLoss
from datasets_triplet import Data

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/root/data/raw', help='Dataset directory')
parser.add_argument('--epoches', type=int, default=80, help='Number of traing epoches')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--init_lr', type=float, default=0.05, help='Initial learning rate')
parser.add_argument('--stride', type=int, default=2, help='Stride for resnet50 in block4')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep  probability)')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight for l2 loss on parameters')
parser.add_argument('--log_interval', type=int, default=1, help='Print iterval')
parser.add_argument('--log_dir', type=str, default='logs', help='Train/val loss and accuracy logs')
parser.add_argument('--checkpoint_interval', type=int, default=10, help='Checkpoint saved interval')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
parser.add_argument('--model', type=str, default='resnet50', help='Model to use')
parser.add_argument('--train_all', action='store_true', help="Use train and val data to train")
parser.add_argument("--random_erasing", action='store_true', help='')
parser.add_argument("--probability", type=float, default=0.5, help='')
parser.add_argument("--batchid", type=int, default=16, help='the batch for id')
parser.add_argument("--batchimage", type=int, default=4, help='the batch of per id')
parser.add_argument("--batchtest", type=int, default=32, help='input batch size for test')
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')
args = parser.parse_args()


func = {'resnet18': resnet50,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'densenet121': densenet121,
        'resnet50_middle': resnet50_middle,
        'osnet': osnet_x1_0
        }

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        #inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss


loader = Data(args)
num_classes = 4768
model = resnet50(num_classes=num_classes, dropout=args.dropout, stride=args.stride)
#model=nn.DataParallel(model,device_ids=[4, 5])
CELoss = CrossEntropyLoss(num_classes)
TripletLoss = TripletLoss()


optimizer = build_optimizer(model, args.init_lr, args.weight_decay)
scheduler = get_scheduler(optimizer)

use_gpu = torch.cuda.is_available()

def train(model):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(args.log_dir)
    for epoch in range(args.epoches + 1):
        starttime = datetime.datetime.now()

        for data in loader.train_loader:
            inputs, labels = data
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
                model = model.cuda()
            features, outputs = model(inputs)
            celoss = CELoss(outputs, labels)
            tripletloss = TripletLoss(features, labels)
            loss = celoss + 2 * tripletloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        train_loss, ce_loss, tri_loss, corrects = 0.0, 0.0, 0.0, 0.0
        for data in loader.train_loader:
            inputs, labels = data
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
                model = model.cuda()
            with torch.no_grad():
                features, outputs = model(inputs)
            celoss = CELoss(outputs, labels)
            ce_loss += celoss
            tripletloss = TripletLoss(features, labels)
            tri_loss += tripletloss
            loss = celoss + 2 * tripletloss
            train_loss += loss
            pred = torch.argmax(outputs, dim=1)
            correct = float(torch.sum(pred == labels))
            corrects += correct
        print(train_loss, ce_loss, tri_loss, corrects)

        #avg_train_loss = train_loss / dataset_sizes[phase]
        #avg_train_ac = train_corrects / dataset_sizes[phase]

        model.train()

        endtime = datetime.datetime.now()
        total_time = (endtime - starttime).seconds

        #writer.add_scalars('loss', {'train_loss': avg_train_loss, 'val_loss': avg_val_loss}, epoch)
        #writer.add_scalars('accuracy', {'train_ac': avg_train_ac, 'val_ac': avg_val_ac}, epoch)
        if epoch % args.log_interval == 0:
            print("="*20, "Epoch {} / {}".format(epoch, args.epoches), "="*20)
            #print("train loss {:.2f}, ce loss {:.2f}, tri loss {:.2f}, train ac {:.2f}".format(avg_train_loss, avg_train_ac))
            #print("val loss {:.2f}, val ac {:.2f}".format(avg_val_loss, avg_val_ac))
            print("lr {:.6f}".format(optimizer.param_groups[0]['lr']))
            print("Training time is {:.2f} s".format(total_time))
            print("\n")
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        if epoch % args.checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "{}_{}.pth".format(args.model, epoch)))


if __name__ == '__main__':
    train(model)

# nohup python -u train.py --data_dir /root/data/Market/pytorch  --model resnet50 &
# nohup python -u train.py --data_dir /root/data/Market/pytorch --model densenet121 &
# nohup python -u train.py --data_dir /root/data/reid_aug --stride 1 --train_all --model resnet50 &
# nohup python -u train.py --data_dir /root/data/Market/pytorch --train_all --model densenet121 &