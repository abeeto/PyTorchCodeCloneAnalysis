import torch
from torch import nn, optim
from torchvision import utils as vutils
from torchvision import transforms
#
from data import datasets
from models import dcgan
import helper

if __name__ == '__main__':
    # setting
    seed = 1340
    batch_size = 32
    workers = 0
    nc = 3
    nz = 100

    lr = 0.0002
    beta1 = 0.5
    start_epoch = 1
    max_epoch = 200
    resume = True

    device = torch.device('cuda')
    torch.manual_seed(seed)

    dataset = datasets.SVHN()
    # dataset = datasets.DogCatData()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=workers, drop_last=True)
    fix_noises = torch.randn(batch_size, nz, 1, 1, device=device) #

    # model and criterion
    netD = dcgan.NetD().to(device)
    netD.apply(helper.weights_init)
    # print(netD)

    netG = dcgan.NetG().to(device)
    netG.apply(helper.weights_init)
    # print(netG)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    if resume:
        check_point = torch.load('checkpoints/dcgan_epoch_100.pt')
        netD.load_state_dict(check_point['netD'])
        netG.load_state_dict(check_point['netG'])
        optimD.load_state_dict(check_point['optimD'])
        optimG.load_state_dict(check_point['optimG'])
        start_epoch = check_point['epoch'] + 1

    netD.train()
    netG.train()
    for epoch in range(start_epoch, max_epoch + 1):
        for i, (imgs, _) in enumerate(data_loader):
            # ====================
            #  1. Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # ====================
            # train with real
            netD.zero_grad()
            real_imgs = imgs.to(device)
            label = torch.full((batch_size,), real_label, device=device) ##

            output = netD(real_imgs)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_imgs = netG(noise)
            label.fill_(fake_label) ##

            output = netD(fake_imgs.detach()) # copy a Variable without grad
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimD.step()


            # ====================
            #  2. Update G network: maximize log(D(G(z)))
            # ====================
            # train with real
            netG.zero_grad()
            label.fill_(real_label) # fake labels are real for generator cost

            output = netD(fake_imgs)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimG.step()

            if i % 20 == 0:
                print('[{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f}   D(G(z)): {:.4f} / {:.4f} / {:.4f}'.format(
                    epoch, max_epoch, i, len(data_loader),
                    errD.item(), errG.item(), D_x, D_G_z2, D_G_z2
                ))

        if epoch % 5 == 0:
            torch.save({'netD': netD.state_dict(), 'netG': netG.state_dict(),
                'epoch': epoch, 'optimD': optimD.state_dict(), 'optimG': optimG.state_dict()
            }, 'checkpoints/dcgan_epoch_{:02d}.pt'.format(epoch))

            fix_fake_imgs = netG(fix_noises)
            helper.plot_imgs(fake_imgs, epoch)
