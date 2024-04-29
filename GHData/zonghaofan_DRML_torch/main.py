import torch
from torch.autograd import Variable
from lib.network import Network
from lib.data_loader import DataSet
import config as cfg
import logging
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import numpy as np
from tensorboardX import SummaryWriter
writer = SummaryWriter('log')

def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
        # print("param_group=",param_group)
        param_group['lr'] = param_group['lr'] * decay_rate
        # print("param_group['lr']=",param_group['lr'])
    return param_group['lr']

logging.basicConfig(level=logging.INFO,
                    format='(%(asctime)s %(levelname)s) %(message)s',
                    datefmt='%d %b %H:%M:%S',
                    filename='logs/region_layer.log',
                    filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('(%(levelname)s) %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

net = Network(cfg.class_number)
#network parameters
# print(net)
if torch.cuda.is_available():
    net.cuda(cfg.cuda_num)

dataset = DataSet(cfg)
#train samples
train_sample_nb = len(dataset.train_dataset)
#batch numbers
train_batch_nb = len(dataset.train_loader)

test_sample_nb = len(dataset.test_dataset)
test_batch_nb = len(dataset.test_loader)

logging.info('Train batch[%d] sample[%d]' % (train_batch_nb, train_sample_nb))
logging.info('Test batch[%d] sample[%d]\n' % (test_batch_nb, test_sample_nb))

#fzh changed
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
opt = optim.Adam(net.parameters(), lr=cfg.lr)
# scheduler = lr_scheduler.CosineAnnealingLR(opt,T_max=5,eta_min=4e-08)

def convert_to_one_hot(class_number, label):
    label = np.eye(class_number)[label.numpy().reshape(-1)].squeeze().astype('uint8')
    label = torch.from_numpy(label)
    if torch.cuda.is_available():
        label=label.cuda(cfg.cuda_num)
    return label

running_loss=0
lr=cfg.lr_decay_rate
for epoch_index in range(cfg.epoch):
    if (epoch_index + 1) % cfg.lr_decay_every_epoch == 0:
        lr=adjust_learning_rate(opt, decay_rate=cfg.lr_decay_rate)

    for batch_index, (img, label) in enumerate(dataset.train_loader):
        print('batch_index=',batch_index)
        label_hot=convert_to_one_hot(cfg.class_number,label.squeeze())
        # print('label_hot',label_hot)
        if torch.cuda.is_available():
            img = img.cuda(cfg.cuda_num)
            label = label.squeeze().cuda(cfg.cuda_num)
        else:
            img = Variable(img)
            label = Variable(label.squeeze())

        pred = net(img)
        # print('pred',pred.shape)
        # print('label',label.data)
        loss = criterion(pred, label)
    #     loss = net.multi_label_sigmoid_cross_entropy_loss(pred, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # scheduler.step()

        statistics_list = net.statistics(pred.data, label_hot.data, cfg.thresh)
        mean_f1_score, f1_score_list = net.calc_f1_score(statistics_list)
        f1_score_list = ['%.4f' % f1_score for f1_score in f1_score_list]

    logging.info('[TRAIN] epoch[%d/%d] loss:%.4f mean_f1_score:%.4f [%s]'
                 % (epoch_index+1, cfg.epoch, loss.item(), mean_f1_score, ' '.join(f1_score_list)))

    writer.add_scalar('Train/Loss', loss.item(), epoch_index+1)
    writer.add_scalar('LR', lr, epoch_index+1)


    if (epoch_index + 1) % cfg.test_every_epoch == 0:
        loss_total = 0
        total_statistics_list = []

        with torch.no_grad():
            for batch_index, (img, label) in enumerate(dataset.test_loader):
                label_hot=convert_to_one_hot(cfg.class_number,label.squeeze())
                # print('label_hot=',label_hot)
                img = Variable(img)
                label = Variable(label.squeeze())

                if torch.cuda.is_available():
                    img = img.cuda(cfg.cuda_num)
                    label = label.squeeze().cuda(cfg.cuda_num)

                pred = net(img)
                # print('pred',pred)
                # print('label',label)
                loss = criterion(pred, label)
                # loss = net.multi_label_sigmoid_cross_entropy_loss(pred, label, size_average=False)
                loss_total += loss

                new_statistics_list = net.statistics(pred.data, label_hot.data, cfg.thresh)
                total_statistics_list = net.update_statistics_list(total_statistics_list, new_statistics_list)

            loss_mean = loss_total / test_sample_nb
            mean_f1_score, f1_score_list = net.calc_f1_score(total_statistics_list)
            f1_score_list = ['%.4f' % f1_score for f1_score in f1_score_list]

            logging.info('[TEST] epoch[%d/%d] loss:%.4f mean_f1_score:%.4f [%s]'
                         % (epoch_index+1, cfg.epoch, loss_mean.item(), mean_f1_score, ','.join(f1_score_list)))
            print('========================================================')
            writer.add_scalar('Test/Loss', loss_mean.item(), epoch_index + 1)
writer.close()


