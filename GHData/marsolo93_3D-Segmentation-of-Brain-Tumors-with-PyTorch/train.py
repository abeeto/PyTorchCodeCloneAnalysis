import torch
import config
from data_loader import *
from loss import *
from model import *
from tqdm import tqdm
import time
from utils import *
import torchsummary

def train(train_loader, model, loss_object, optimizer, scaler, current_step, global_steps, epoch):
    # if current_step == 0:
    #     load_checkpoint(config.LOAD_CHECKPOINT + '.pth.tar', model, optimizer, optimizer.param_groups[0]['lr'])

    loop = tqdm(train_loader, leave=True)
    losses = []
    ETs = []
    EDs = []
    WTs = []
    dices = []
    start_time = time.time()
    for i, (x, y) in enumerate(loop):
        if current_step == 0:
            end_generating_time = time.time()
            print('Generation Time: ', end_generating_time - start_time)

        optimizer.zero_grad()
        if current_step < config.WARMUP_STEPS:
            optimizer.param_groups[0]['lr'] = ((config.LEARNING_RATE - config.LEARING_RATE_BEGIN) / config.WARMUP_STEPS) * current_step + config.LEARING_RATE_BEGIN
        else:
            optimizer.param_groups[0]['lr'] = torch.tensor(config.LEARNING_RATE_END + 0.5 * (config.LEARNING_RATE - config.LEARNING_RATE_END) * (
                (1 + torch.cos((current_step) / (global_steps) * np.pi))))

        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        if current_step == 0:
            start_time = time.time()
        pred = model(x.float())
        if current_step == 0:
            end_time = time.time()
            print('NN Calculation Time: ', end_time - start_time)

        dice, iou, lossiou = loss_object(pred, y.float())

        losses.append(lossiou.item())
        ETs.append(iou.cpu().detach().numpy()[1])
        EDs.append(iou.cpu().detach().numpy()[2])
        WTs.append(iou.cpu().detach().numpy()[0])

        dices.append(dice.item())

        if current_step == 0:
            start_time = time.time()
        optimizer.zero_grad()
        dice.backward()
        optimizer.step()
        if current_step == 0:
            end_time = time.time()
            print('BackProp Time: ', end_time - start_time)

        mean_loss = sum(losses) / len(losses)
        mean_et = sum(ETs) / len(ETs)
        mean_ed = sum(EDs) / len(EDs)
        mean_wt = sum(WTs) / len(WTs)
        mean_iou = (mean_ed + mean_et + mean_wt) / 3
        mean_dice = sum(dices) / len(dices)

        current_step = current_step + 1

        loop.set_postfix(loss=mean_loss, lr=optimizer.param_groups[0]['lr'].item(), current_step=current_step, ET=mean_et, ED=mean_ed, WT=mean_wt, iou=mean_iou, dice=mean_dice)

    save_checkpoint(model, optimizer, filename=config.CHECKPOINT + '_' + str(epoch) + '.pth.tar')

    return current_step

def test(test_loader, model, loss_object):
    loop = tqdm(test_loader, leave=True)

    losses = []
    ETs = []
    EDs = []
    WTs = []
    #NONs = []
    dices = []

    for i, (x, y) in enumerate(loop):
        if i == 20:
            break
        #x = x[:, 0:1, ...]
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        with torch.no_grad():
            pred = model(x.float())
            dice, iou, lossiou = loss_object(pred, y)

        losses.append(lossiou.item())
        ETs.append(iou.cpu().detach().numpy()[2])
        EDs.append(iou.cpu().detach().numpy()[1])
        WTs.append(iou.cpu().detach().numpy()[0])
        dices.append(dice.item())

        mean_loss = sum(losses) / len(losses)
        mean_et = sum(ETs) / len(ETs)
        mean_ed = sum(EDs) / len(EDs)
        mean_wt = sum(WTs) / len(WTs)
        #mean_non = sum(NONs) / len(NONs)
        mean_iou = (mean_ed + mean_et + mean_wt) / 3
        mean_dice = sum(dices) / len(dices)

        loop.set_postfix(loss=mean_loss, ET=mean_et, ED=mean_ed, WT=mean_wt, iou=mean_iou, dice=mean_dice)

def main():
    global current_step

    model = UNet3D(4, [16, 32, 64, 128, 256], num_classes=config.NUM_CLASSES).to(config.DEVICE) # [8, 16, 32, 64] für Größe von 184x184x184

    model = model.float()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    loss_object = LossAndMetric(num_classes=config.NUM_CLASSES)

    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader = dataset_loader()

    global_steps = torch.tensor(len(train_loader) * config.EPOCHS)
    current_step = torch.tensor(0)

    for epoch in range(config.EPOCHS):
        print('####################### EPOCH ' + str(epoch+1) + ' ###########################')
        print('TRAINING')
        print(current_step)
        current_step = train(train_loader, model, loss_object, optimizer, scaler, current_step, global_steps, epoch)
        print('TEST')
        test(train_loader, model, loss_object)


if __name__ == '__main__':
    main()

