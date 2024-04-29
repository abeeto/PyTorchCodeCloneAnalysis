import os
from datetime import datetime
import random
import numpy as np
from tqdm import tqdm
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from args_file import set_args
from Dataset.dataset import AVADataset, train_transform, val_transform
from network import NIMA_Dict
from EMD_Loss import emd_loss
import warnings
warnings.filterwarnings("ignore")

# 固定随机种子
def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    # 获取计算设备
    seed_torch(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-当前计算设备：{}".format(torch.cuda.get_device_name(0)))

    # 导入数据集
    train_set = AVADataset(
        csv_file_path=args.train_csv_file,
        image_path=args.image_path,
        transform=train_transform,
        image_num=1000
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    # 训练一定步数时使用的验证集：500张
    val_set_step = AVADataset(
        csv_file_path=args.val_csv_file,
        image_path=args.image_path,
        transform=val_transform,
        image_num=500
    )
    val_loader_step = DataLoader(
        val_set_step,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    # 训练一个epoch时使用的验证集：约12000
    val_set_epoch = AVADataset(
        csv_file_path=args.val_csv_file,
        image_path=args.image_path,
        transform=val_transform
    )
    val_loader_epoch = DataLoader(
        val_set_epoch,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print('-训练数据与验证数据导入完成')

    # 构建神经网络
    net = NIMA_Dict[args.network].to(device)
    print("-NIMA_{} 构建完成，参数量为： {} ".format(args.network, sum(x.numel() for x in net.parameters())))

    # 设定优化器
    optimizer = optim.SGD([
        {'params': net.features.parameters(), 'lr': args.conv_base_lr},
        {'params': net.classifier.parameters(), 'lr': args.linear_lr}],
        momentum=0.9
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)

    # 继续训练之前的权重
    start_epoch = 0
    if args.warm_start_path is not None:
        checkpoint = torch.load(args.warm_start_path)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print('-载入完成')

    # 训练记录
    train_time = datetime.now().strftime("%m-%d_%H-%M")
    logs_name = args.network + '_' + train_time + '_epoch{}'.format(args.epochs + start_epoch)
    logs_dir = './logs/' + logs_name
    writer = SummaryWriter(logs_dir)
    print('-日志保存路径：' + logs_dir)
    print('--使用该指令查看训练过程：tensorboard --logdir=./')

    # 保存权重的主路径
    save_path = os.path.join(args.checkpoint_path, logs_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 开始训练
    print('-开始训练...')
    step = 0  # 参数更新次数
    init_val_loss = float('inf')
    stop_count = 0
    for epoch in range(start_epoch, args.epochs + start_epoch):
        loop = tqdm(train_loader)
        for i, sample_batch in enumerate(loop):
            images = sample_batch['image'].to(device)
            labels = sample_batch['label'].to(device)
            outputs = net(images)

            optimizer.zero_grad()
            loss = emd_loss(labels, outputs)
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch + 1}/{args.epochs + start_epoch}]")
            loop.set_postfix(
                loss=loss.item(),
            )

            step += 1
            # 每迭代100次记录一次训练损失
            if step % 100 == 0:
                writer.add_scalar('train_loss', loss.item(), global_step=step)
            # 每迭代500次记录一次验证损失
            if step % 500 == 0:
                net.eval()
                loss_list = []
                for j, sample in enumerate(val_loader_step):
                    images = sample['image'].to(device)
                    labels = sample['label'].to(device)
                    with torch.no_grad():
                        outputs = net(images)
                    loss = emd_loss(labels, outputs)
                    loss_list.append(loss.item())
                mean_loss = sum(loss_list) / len(loss_list)

                writer.add_scalar('val_step_emd_loss', mean_loss, global_step=epoch+1)
                net.train()

        # 每完成一个epoch完整验证一次
        print('--第{}个epoch完成，开始验证...'.format(epoch + 1))
        net.eval()
        val_loss_list = []
        for j, sample in enumerate(val_loader_epoch):
            images = sample['image'].to(device)
            labels = sample['label'].to(device)
            with torch.no_grad():
                outputs = net(images)
            loss = emd_loss(labels, outputs)
            val_loss_list.append(loss.item())
        val_loss = sum(val_loss_list) / len(val_loss_list)

        writer.add_scalar('val_epoch_emd_loss', val_loss, global_step=step)
        net.train()
        print('---验证完成，val_loss={}'.format(val_loss))

        # 如果当前val_loss比上一个epoch小，保存当前权重
        # 如果连续3个epoch的val_loss不下降，提前终止训练
        if val_loss < init_val_loss:
            init_val_loss = val_loss
            print('--保存本轮权重...')

            save_name = os.path.join(save_path + '/',
                                     'epoch{}_{}.pt'.format(args.epochs + start_epoch, epoch + 1 + start_epoch))
            torch.save(
                {
                    'net_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': (args.epochs + start_epoch)
                }, save_name
            )
            stop_count = 0
        elif val_loss >= init_val_loss:
            stop_count += 1
            if stop_count == args.stop_count:
                break

        # 学习率记录
        scheduler.step()
        writer.add_scalar('cnn_lr', optimizer.param_groups[0]['lr'], global_step=epoch + 1)
        writer.add_scalar('linear_lr', optimizer.param_groups[1]['lr'], global_step=epoch + 1)

    # 最后一轮的权重也保存
    save_name = os.path.join(save_path + '/', 'final_epoch{}.pt'.format(args.epochs))
    torch.save(
        {
            'net_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': (args.epochs + start_epoch)
        }, save_name
    )
    print('-训练完成，权重文件的路径为：' + save_path)


if __name__ == '__main__':
    args = set_args()
    main(args)
