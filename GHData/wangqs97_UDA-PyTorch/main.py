import torch
import torch.nn as nn
from torch import optim
from load_data import dataload
from tsa import TSA
from resnet import ResNet18
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import argparse
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--schedule")
parser.add_argument("--epoch")
parser.add_argument("--tsa", default=1)
parser.add_argument("--supnum", type=int, default=4000)
args = parser.parse_args()

tb_logger = SummaryWriter(log_dir=f'./tb_logger/{args.schedule}{args.epoch}')

def train():
    total_loss = 0.
    thres = -1
    for i, unsup_data in enumerate(tqdm(unsup_dataloader)):
        net.train()
        step = epoch * len(unsup_dataloader) + i
        # supervised training
        if i % 6 == 5:
            try:
                sup_img, label = next(sup_iter)
            except:
                sup_iter = iter(sup_dataloader)
                sup_img, label = next(sup_iter)
                
            sup_img = sup_img.to(DEVICE)
            label = label.to(DEVICE)
            label_pred = net(sup_img).to(DEVICE)
            label_prob = torch.softmax(label_pred, dim=-1)
            sup_loss = sup_criterion(label_pred, label)

            # TSA
            if tsa_enable == '1':
                thres, avg_sup_loss = TSA(label_prob, label, sup_loss, step, schedule=args.schedule, \
                                                                        total_step=total_step, start=0.4)
                
            else:
                avg_sup_loss = sup_loss

            total_loss += avg_sup_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            total_loss = 0.

            tb_logger.add_scalar("sup_loss", avg_sup_loss.item(), step)
            tb_logger.add_scalar("TSA_thresh", thres, step)
            tb_logger.add_scalar("cos_lr", scheduler.get_lr()[0], step)

        # unsupervised training
        aug_unsup_img, unsup_img = unsup_data
        unsup_img = unsup_img.to(DEVICE)
        aug_unsup_img = aug_unsup_img.to(DEVICE)

        unsup_label_pred = net(unsup_img).detach()
        unsup_label_prob = torch.softmax(unsup_label_pred, dim=-1)
        aug_unsup_label_pred = net(aug_unsup_img)
        aug_unsup_label_prob = torch.log_softmax(aug_unsup_label_pred, dim=-1)
        unsup_loss = unsup_criterion(aug_unsup_label_prob, unsup_label_prob)
        total_loss += unsup_loss

        tb_logger.add_scalar("unsup_loss", unsup_loss.item(), epoch)


def val():
    correct = 0
    net.eval()
    with torch.no_grad():
        for _, data_eval in enumerate(val_dataloader):
            img_eval, label_eval = data_eval
            img_eval = img_eval.to(DEVICE)
            label_eval = label_eval.to(DEVICE)
            eval_pred = net(img_eval).to(DEVICE)
            val_loss = val_criterion(eval_pred, label_eval)
            
            tb_logger.add_scalar("val_loss", val_loss.item(), epoch)

            correct += (torch.max(eval_pred, dim=-1).indices == label_eval).sum().data
    correct_rate = correct.float() / len(val_dataloader) / 16 * 100
    print('epoch: {}, correct number: {}, correct rate: {:.2f}%'.format(epoch + 1, correct, correct_rate))

    tb_logger.add_scalar("val_accuracy", correct_rate.item(), epoch)

    torch.save({'params': net.state_dict(), 'epoch': epoch + 1}, f'./model/{model_name}.pth')


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = f'{args.schedule}{args.epoch}'
    tsa_enable = args.tsa
    total_epoch = int(args.epoch)
    net = ResNet18().to(DEVICE)
    if not os.path.exists(f'./model'):
        os.mkdir('model')
    if os.path.exists(f'./model/{model_name}.pth'):
        load = torch.load(f'./model/{model_name}.pth')
        current_epoch = load['epoch']
        net.load_state_dict(load['params'])
        sup_dataloader, val_dataloader, unsup_dataloader = dataload(1, args.supnum)
    else:
        current_epoch = 0
        sup_dataloader, val_dataloader, unsup_dataloader = dataload(0, args.supnum)
    total_step = total_epoch * len(unsup_dataloader)
    sup_criterion = nn.CrossEntropyLoss(reduction='none') if tsa_enable == '1' else nn.CrossEntropyLoss()
    val_criterion = nn.CrossEntropyLoss()
    unsup_criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam([{'params': net.parameters(), 'initial_lr': 3e-3}])
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epoch, eta_min=0, last_epoch=current_epoch)
    
    for epoch in range(current_epoch, total_epoch):
        train()
        val()
        scheduler.step()
    
    tb_logger.close()
    os.remove('temp_label.pkl')
        
