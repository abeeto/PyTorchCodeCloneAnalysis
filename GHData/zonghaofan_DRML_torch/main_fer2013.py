import torch
from torch.autograd import Variable
from network_fer2013_deep_short import Network_new
from data_loader_fer2013_3channels import FER2013
import config as cfg
import logging
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import numpy as np
import torch.utils.data as data
from tensorboardX import SummaryWriter
import os
writer = SummaryWriter('log_fer2013_deep_short_3channels_48')

# def adjust_learning_rate(optimizer, decay_rate=.9):
#     for param_group in optimizer.param_groups:
#         # print(param_group['lr'])
#         # print("param_group=",param_group)
#         param_group['lr'] = param_group['lr'] * decay_rate
#         # print("param_group['lr']=",param_group['lr'])
#         # print(type(param_group['lr']))
#     return param_group['lr']

def ajust_learning_tri(optimizer,clr_iterations,step_size,base_lr=1e-5, max_lr=1e-3):
    cycle = np.floor(1 + clr_iterations / (2 * step_size))
    x = np.abs(clr_iterations / step_size - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) /(2 ** (cycle - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] =lr
    return lr

# logging.basicConfig(level=logging.INFO,
#                     format='(%(asctime)s %(levelname)s) %(message)s',
#                     datefmt='%d %b %H:%M:%S',
#                     filename='logs/region_layer.log',
#                     filemode='w')

# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# formatter = logging.Formatter('(%(levelname)s) %(message)s')
# console.setFormatter(formatter)
# logging.getLogger('').addHandler(console)
print("start...")
model = Network_new(cfg.class_number)

if torch.cuda.is_available():
    model.cuda(cfg.cuda_num)
#train samples
train_data=FER2013(path='./data/fer2013/data/fer2013_new_data.h5',split='Training')
train_loader = data.DataLoader(dataset=train_data,
                               batch_size=cfg.batch_size,
                               shuffle=True)
train_batch_nb=len(train_data)

#test samples
test_data=FER2013(path='./data/fer2013/data/fer2013_new_data.h5',split='Testing')
test_loader = data.DataLoader(dataset=test_data,
                               batch_size=cfg.batch_size,
                               shuffle=True)
test_batch_nb=len(test_data)

logging.info('Train  sample[%d]' % (train_batch_nb))
logging.info('Test  sample[%d]\n'% (test_batch_nb))

#fzh changed
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
# opt=optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
opt = optim.Adam(model.parameters(), lr=cfg.lr,weight_decay=1e-3)

checkpoint = {'model': model,
              'state_dict': model.state_dict(),
              'optimizer': opt.state_dict()}

def convert_to_one_hot(class_number, label):
    label = np.eye(class_number)[label.numpy().reshape(-1)].squeeze().astype('uint8')
    label = torch.from_numpy(label)
    if torch.cuda.is_available():
        label=label.cuda(cfg.cuda_num)
    return label

# def clip_gradient(optimizer, grad_clip):
#     for group in optimizer.param_groups:
#         #print(group['params'])
#         for param in group['params']:
#             param.grad.data.clamp_(-grad_clip, grad_clip)

if os.path.exists(cfg.model_path):
    checkpoint = torch.load(cfg.model_path + '/model.pth')
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
else:
    print('no model')
    os.mkdir(cfg.model_path)

running_loss=0
max_acc=0
for epoch_index in range(cfg.epoch):
    model.train()
    index_train = epoch_index * (train_batch_nb//cfg.batch_size+1)
    index_test = epoch_index * (test_batch_nb//cfg.batch_size+1)
    for batch_index, (img, label) in enumerate(train_loader):
        batch_index+=index_train
        lr=ajust_learning_tri(opt,batch_index,step_size=8*len(train_data)//cfg.batch_size)
        # print('batch_index=',batch_index)
        label=label.view(-1,1)
        label_hot=convert_to_one_hot(cfg.class_number,label.squeeze())

        img = Variable(img)
        label = Variable(label)

        if torch.cuda.is_available():
            img = img.cuda(cfg.cuda_num)
            label = label.squeeze().cuda(cfg.cuda_num)

        opt.zero_grad()

        pred = model(img)
        # print(pred.shape)
        predicted= torch.argmax(pred, 1)
      #  print('label: ', label)
        # print(predicted.shape)
        acc=(predicted == label.squeeze()).sum().float()/len(label)
        # print('acc:',acc)
        loss = criterion(pred, label)

        loss.backward()
        # clip_gradient(opt, 0.1)
        opt.step()

        statistics_list = model.statistics(pred.data, label_hot.data, cfg.thresh)
        mean_f1_score, f1_score_list = model.calc_f1_score(statistics_list)
        f1_score_list = ['%.4f' % f1_score for f1_score in f1_score_list]

        writer.add_scalar('Train/Loss', loss.item(), (batch_index + 1) * (epoch_index + 1))
        writer.add_scalar('Train/Acc', acc.item(), (batch_index + 1) * (epoch_index + 1))
        writer.add_scalar('LR', lr, (batch_index + 1) * (epoch_index + 1))

    logging.info('[TRAIN] epoch[%d/%d] loss:%.4f mean_f1_score:%.4f [%s]'
                 % (epoch_index+1, cfg.epoch, loss.item(), mean_f1_score, ' '.join(f1_score_list)))


    with torch.no_grad():
        model.eval()
        aver_acc_list=[]
        for batch_index, (img, label) in enumerate(test_loader):
            batch_index += index_test
            label=label.view(-1,1)
            label_hot=convert_to_one_hot(cfg.class_number,label.squeeze())

            img = Variable(img)
            label = Variable(label.squeeze())

            if torch.cuda.is_available():
                img = img.cuda(cfg.cuda_num)
                label = label.squeeze().cuda(cfg.cuda_num)

            pred = model(img)
            predicted = torch.argmax(pred, 1)
            # print(predicted.shape)
            acc = (predicted == label.squeeze()).sum().float() / len(label)
            aver_acc_list.append(acc.cpu().numpy())
            # print('acc:', acc)
            loss = criterion(pred, label)
            new_statistics_list = model.statistics(pred.data, label_hot.data, cfg.thresh)
            mean_f1_score, f1_score_list = model.calc_f1_score(new_statistics_list)
            f1_score_list = ['%.4f' % f1_score for f1_score in f1_score_list]
            writer.add_scalar('Test/Loss', loss.item(), (batch_index + 1)*(epoch_index + 1))
            writer.add_scalar('Test/Acc', acc.item(), (batch_index + 1) * (epoch_index + 1))

        logging.info('[TEST] epoch[%d/%d] loss:%.4f mean_f1_score:%.4f [%s]'
                     % (epoch_index+1, cfg.epoch, loss.item(), mean_f1_score, ','.join(f1_score_list)))
        print('========================================================')
        if np.array(aver_acc_list).mean()>max_acc:
            max_acc=np.array(aver_acc_list).mean()
            print('max_acc=',max_acc)
    # if epoch_index%10==0:
        # torch.save(model.state_dict(), cfg.model_path+'/model_{}.pth'.format(epoch_index))
            torch.save(checkpoint, cfg.model_path+'/model.pth')

writer.close()

