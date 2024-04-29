import os
import torch
import argparse
from models import creat_model
from cv2dataset import CV2Dataset
from torch.utils.data import DataLoader


def train(opt):
    path_train, path_val, model_dir, model_end, model_last = \
        opt.path_train, opt.path_val, opt.model_dir, opt.model_end, opt.model_last
    batch_size, workers, img_size, lr, start_epoch, epochs, eval_epoch = \
        opt.batch_size, opt.workers, opt.img_size, opt.lr, opt.start_epoch, opt.epochs, opt.eval_epoch
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    # 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('mps')    # mac M1使用'mps' or 'cpu
    # 定义数据集
    train_data = CV2Dataset(path_train, img_size)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_data = CV2Dataset(path_val, img_size)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=nw)
    # 定义网络
    model = creat_model(opt.name).to(device)
    # 定义优化器
    optim = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.5, 0.999))
    # 定义损失函数
    loss_fun = torch.nn.BCELoss().to(device)
    # 是否恢复训练
    model_resume = os.path.join(model_dir, model_last)
    print(model_resume)
    best_acc = 0
    if opt.resume and os.path.exists(model_resume):
        ckp = torch.load(model_resume)
        if len(ckp.keys()) > 1:
            best_acc = ckp['best_acc']
            start_epoch = ckp['epoch'] + 1
            model.load_state_dict(ckp['model'])
            optim.load_state_dict(ckp['optimizer'])
            if start_epoch > epochs:
                print(f'训练已完成，训练次数{start_epoch}/{epochs}，无需恢复训练')
                return
        elif len(ckp.keys()) == 1:
            model.load_state_dict(ckp['model'])
        else:
            print('文件已损坏或训练已完成，无法恢复训练')
            return
    # 开始训练
    batch_count = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for i, (indata, label) in enumerate(train_loader):
            indata = indata.to(device)
            label = label.to(device)
            optim.zero_grad()
            pred = model(indata)
            loss = loss_fun(pred, label)
            loss.backward()
            optim.step()
            train_loss_sum += loss.item()
            train_acc_sum += sum(row.all().int().item() for row in (pred.ge(0.5) == label))
            n += indata.shape[0]
            batch_count += 1
            # print(f'epoch: {epoch}; step: {i}; loss: {loss}')
        print(f'epoch: {epoch}; loss: {train_loss_sum / batch_count}; acc: {train_acc_sum / n}')
        if (epoch + 1) % eval_epoch == 0:
            acc = val(opt, model, val_loader)
            checkpoint = {
                'best_acc': acc,
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optim.state_dict()}
            model_save_name = 'model_last.pt'
            model_save_path = os.path.join(model_dir, model_save_name)
            torch.save(checkpoint, model_save_path)
            if best_acc <= acc:
                best_acc = acc
                model_save_name = 'model_best.pt'
                model_save_path = os.path.join(model_dir, model_save_name)
                torch.save(checkpoint, model_save_path)
    torch.save({'model': model.state_dict()}, os.path.join(model_dir, model_end))


def val(opt, model=None, val_loader=None):
    path_val, model_dir, model_last = opt.path_val, opt.model_dir, opt.model_last
    batch_size, workers, img_size = opt.batch_size, opt.workers, opt.img_size
    if model is not None and val_loader is not None:
        device = next(model.parameters()).device
    else:
        model_path = os.path.join(model_dir, model_last)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('mps')    # mac M1使用'mps' or 'cpu
        model = creat_model(opt.name).to(device)
        model.load_state_dict(torch.load(model_path)['model'])
        nd = torch.cuda.device_count()  # number of CUDA devices
        nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
        val_data = CV2Dataset(path_val, img_size)
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=nw)
    model.eval()
    val_acc_sum, n = 0.0, 0
    with torch.no_grad():
        for i, (indata, label) in enumerate(val_loader):
            indata = indata.to(device)
            label = label.to(device)
            pred = model(indata)
            val_acc_sum += sum(row.all().int().item() for row in (pred.ge(0.5) == label))
            n += indata.shape[0]
    acc = val_acc_sum / n
    print(f'acc: {acc}')
    return acc


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_train', type=str, default='E:/DataSources/DogsAndCats/train', help='')
    parser.add_argument('--path_val', type=str, default='E:/DataSources/DogsAndCats/val', help='')
    parser.add_argument('--name', type=str, default='ViT', help='VGG11, ViT, Swin')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--model_dir', type=str, default='./weights')
    parser.add_argument('--model_end', type=str, default='model_end.pt')
    parser.add_argument('--model_last', type=str, default='model_best.pt')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--eval_epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--img_size', type=int, default=224, help='train, val image size (pixels)')
    parser.add_argument('--batch_size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu or mps')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == '__main__':
    option = parse_opt(True)
    train(option)
    # val(option)
