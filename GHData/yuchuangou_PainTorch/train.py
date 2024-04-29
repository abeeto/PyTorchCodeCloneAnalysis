import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from network import define_D, GanLoss, define_U
from dataloader import get_train_loader
from config import CFG


cudnn.benchmark = True

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)

device = torch.device('cuda') if CFG.cuda else torch.device('cpu')

for path in [CFG.result_dir, CFG.model_dir]:
    if not os.path.isdir(path):
        os.mkdir(path)

print('Setup ImageLoader')
loader = get_train_loader(CFG.color_path,
                          CFG.batch_size, resize=True, size=(512, 512))

print('Define Network')
netG = define_U(device, in_channel=1, out_channel=3).to(device)
netD = define_D(device, 3).to(device)

print('Define Optimizer')
optG = optim.Adam(netG.parameters(), lr=CFG.lr, betas=CFG.betas)
optD = optim.Adam(netD.parameters(), lr=CFG.lr, betas=CFG.betas)

if CFG.is_load_model:
    try:
        saved = torch.load(os.path.join(CFG.model_dir, CFG.model_name))
        start_epoch = saved['epoch']
        netG.load_state_dict(saved['netG'])
        netD.load_state_dict(saved['netD'])
        optG.load_state_dict(saved['optG'])
        optD.load_state_dict(saved['optD'])
        print('succeed to load model')
    except:
        print('Failed to load model.')

print('Setup Loss function')
criterionGAN = GanLoss()
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()

fake_label = 0
real_label = 1


def train(epoch):
    for i, (line, color) in enumerate(loader, 1):
        line = line.to(device)
        color = color.to(device)

        fake = netG(line)

        ####################
        # Update D Network #
        ####################
        optD.zero_grad()
        # Fake
        pred_fake = netD(fake.detach())
        loss_d_fake = criterionGAN(pred_fake, fake_label)

        # Real
        pred_real = netD(color)
        loss_d_real = criterionGAN(pred_real, real_label)

        loss_d = (loss_d_fake + loss_d_real) * 0.5
        loss_d.backward()
        optD.step()

        ###################
        # UpdateG Network #
        ###################
        optG.zero_grad()
        pred_fake = netD(fake)

        loss_g_gan = criterionGAN(pred_fake, real_label)
        loss_g_l1 = criterionL1(fake, color)
        loss_g_mse = criterionMSE(fake, color)
        loss_g = loss_g_gan + (loss_g_l1 + loss_g_mse) * 10

        loss_g.backward()
        optG.step()

        if i % CFG.loss_span == 0:
            print(f'Epoch: {epoch}, Ep.{i},'
                  f' Loss_D: {loss_d.item():.5f},'
                  f' LossG: {loss_g.item():.5f}')

        if i % CFG.save_result_span == 0:
            vutils.save_image(
                torch.cat([color, fake]),
                os.path.join(CFG.result_dir, f'{epoch}_{i}.jpg'),
                normalize=True,
                nrow=4
            )

        if i % CFG.save_model_span == 0:
            torch.save({
                'epoch': epoch,
                'netG': netG.state_dict(),
                'netD': netD.state_dict(),
                'optG': optG.state_dict(),
                'optD': optD.state_dict()
            }, os.path.join(CFG.model_dir, CFG.model_name))


if __name__ == '__main__':
    print('\nStart Train...')
    for e in range(CFG.start_epoch, CFG.num_epoch+1):
        train(e)
