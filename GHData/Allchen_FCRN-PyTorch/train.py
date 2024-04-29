import os
import argparse
import path

import torch
import visdom
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable

import nyud_dataset
import models.FCRN as FCRN
import models.upsample as upsample


def main(
        use_gpu=True, batch_size=32, epoch_num=20,
        start_epoch=0, learning_rate=1e-3,
        load_checkpoint=None, load_file=None, save_rate=10
        ):
    print('FCRN')
    if use_gpu and not torch.cuda.is_available():
        print('Cannot use GPU')
        use_gpu = False

    load_ready = False
    if load_checkpoint is not None:
        print('Loading checkpoint from epoch {0:d}'.format(load_checkpoint))
        load_network_path = \
            'cache/network/FCRN_e{0:d}.pth'.format(load_checkpoint)
        load_optimizer_path = \
            'cache/optimizer/optimizer_e{0:d}.pth'.format(load_checkpoint)
        if os.path.exists(load_network_path)\
           and os.path.exists(load_optimizer_path):
            start_epoch = load_checkpoint
            load_ready = True
        else:
            msg = 'Unable to load files:\n'
            if not os.path.exists(load_network_path):
                msg += load_network_path
            if not os.path.exists(load_optimizer_path):
                msg += load_optimizer_path
            print(msg)
            return False

    elif load_file is not None:
        load_network_path = load_file[0]
        load_optimizer_path = load_file[1]
        if os.path.exists(load_network_path)\
           and os.path.exists(load_optimizer_path):
            load_ready = True
        else:
            msg = 'Unable to load files:\n'
            if not os.path.exists(load_network_path):
                msg += load_network_path
            if not os.path.exists(load_optimizer_path):
                msg += load_optimizer_path
            print(msg)
            return False

    vis = visdom.Visdom()
    win_rgb = vis.image(
        torch.zeros(240, 320), opts=dict(title='GT'), env='FCRN'
        )
    win_Depth = vis.image(
        torch.zeros(240, 320), opts=dict(title='DG'), env='FCRN'
        )
    win_Pred = vis.image(
        torch.zeros(240, 320), opts=dict(title='Predicted'), env='FCRN'
        )
    win_Training_Loss = vis.image(
        torch.zeros(3, 450, 450), opts=dict(title='Refinement loss'),
        env='FCRN'
        )
    win_Epoch_Loss = vis.image(
        torch.zeros(3, 450, 450), opts=dict(title='Refinement loss'),
        env='FCRN'
        )

    print('Initializing network...')
    fcrn = FCRN.FCRN(up_conv=upsample.UpProjection2d, use_gpu=use_gpu)
    optimizer = torch.optim.Adam(
        fcrn.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999)
        )
    if load_ready:
        fcrn.load_state_dict(torch.load(load_network_path))
        optimizer.load_state_dict(torch.load(load_optimizer_path))
        msg = 'Successfully loaded parameter from '
        msg += 'checkpoint {0:d}'.format(load_checkpoint)
        print(msg)
    loss_function = torch.nn.MSELoss()
    if use_gpu:
        loss_function = loss_function.cuda()

    os.system('mkdir -p cache/network/ cache/optimizer/')
    training_losses = []
    epoch_losses = []
    for epoch_i in tqdm(range(start_epoch, start_epoch+epoch_num)):
        train_data_loader = \
            nyud_dataset.get_nyud_train_set(
                    (228, 304), batch_size=batch_size, data_augmentation=True
                    )
        train_data_iter = iter(train_data_loader)
        epoch_loss = 0
        for batch_i in tqdm(range(len(train_data_iter))):
            data = train_data_iter.next()
            inputs, gts, inputs_disp = data
            inputs = Variable(inputs)
            gts = Variable(gts)
            gts = torch.nn.functional.upsample(
                gts, size=(128, 160), mode='bilinear'
                )
            if use_gpu:
                inputs = inputs.cuda()
                gts = gts.cuda()

            outputs = fcrn(inputs)

            loss = loss_function(outputs, gts)
            loss.backward()
            optimizer.step()
            loss = loss.cpu().data[0]
            epoch_loss += loss

            training_losses.append(loss)
            gts_disp = gts / gts.max()
            outputs_disp = outputs / outputs.max()
            if len(training_losses) > 1000:
                training_losses = training_losses[1:]
            vis.image(
                inputs_disp[0, :, :, :].cpu(), opts=dict(title='RGB'),
                env='FCRN', win=win_rgb
                )
            vis.image(
                gts_disp.data[0, :, :, :].cpu(), opts=dict(title='Depth'),
                env='FCRN', win=win_Depth
                )
            vis.image(
                outputs_disp.data[0, :, :, :].cpu(),
                opts=dict(title='Prediction'),
                env='FCRN', win=win_Pred
                )
            vis.line(
                np.array(training_losses), np.arange(len(training_losses)),
                opts=dict(title='Training Loss'), env='FCRN',
                win=win_Training_Loss
                )
        epoch_loss /= len(train_data_iter)
        epoch_losses.append(epoch_loss)
        vis.line(
            np.array(epoch_losses), np.arange(len(epoch_losses)),
            opts=dict(title='Epoch Loss'), env='FCRN',
            win=win_Epoch_Loss
            )
        if epoch_i % save_rate == 0:
            save_path = 'cache/network/'
            save_path += 'FCRN_e{0:d}.pth'.format(epoch_i)
            torch.save(fcrn.state_dict(), save_path)
            save_path = 'cache/optimizer/'
            save_path += 'optimizer_e{0:d}.pth'.format(epoch_i)
            torch.save(optimizer.state_dict(), save_path)

    epoch_losses_file = open('cache/epoch_losses.txt', 'w')
    epoch_losses_file.write(str(epoch_losses))
    epoch_losses_file.close()

    return True


if __name__ == '__main__':
    configures = argparse.ArgumentParser()
    configures.add_argument(
        '--cuda', required=False, action='store_true'
        )
    configures.add_argument(
        '--batch_size', required=False, type=int, default=16
        )
    configures.add_argument(
        '--epoch_num', required=False, type=int, default=500
        )
    configures.add_argument(
        '--learning_rate', required=False, type=float, default=1e-4
        )
    configures.add_argument(
        '--load_checkpoint', required=False, type=int, default=None
        )
    configures.add_argument(
        '--load_file', required=False, type=str, default=None
        )
    configures.add_argument(
        '--start_epoch', required=False, type=int, default=0
        )
    configures.add_argument(
        '--save_rate', required=False, type=int, default=10
        )
    configures = configures.parse_args()

    main(
        use_gpu=configures.cuda, batch_size=configures.batch_size,
        epoch_num=configures.epoch_num,
        start_epoch=configures.start_epoch,
        learning_rate=configures.learning_rate,
        load_checkpoint=configures.load_checkpoint,
        load_file=configures.load_file,
        save_rate=configures.save_rate
        )
