import os
from collections import defaultdict

import argparse

import pprint

from config.config import get_params
from generate_noise import get_noise, get_noiseByContinuous_code
from gpu_utils import auto_select_gpu
from models import prepare_model

os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()
import torch.optim as optim
import torchvision.utils as vutils
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataloader import get_data
from utils import *

parser = argparse.ArgumentParser('InfoGAN')
parser.add_argument('--dataset', dest='dataset', help='Training dataset', default='faces', type=str)
parser.add_argument('--output_folder', dest='output_folder', help='the dir save result', default='output1', type=str)
parser.add_argument('--comment', dest='comment', help='comment', default='_change', type=str)
args = parser.parse_args()

print('Args parameters')
pprint.pprint(vars(args))

params = get_params(args.dataset)
# Set random seed for reproducibility.

random.seed(params.seed)
np.random.seed(params.seed)
torch.manual_seed(params.seed)

# Use GPU if available.
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")


def plot_training_image(dataloader, writer):
    '''
    Plot the training images.

    :param dataloader: show example image
    :param writer:
    :return:
    '''
    print('Draw sample image!')
    sample_batch = next(iter(dataloader))
    grid = vutils.make_grid(sample_batch[0].to(device)[: 100], nrow=10, padding=2, normalize=True).cpu()
    writer.add_image('example', grid, 0)
    print('Draw sample image, finish!')

    # pass


def train(dataloader, writer):
    netG, netQ, netD, discriminator = prepare_model(params, device)

    # -----------------------Prepare loss and optimizer-----------------------------
    # Loss for discrimination between real and fake images.
    criterionD = nn.BCELoss()
    # Loss for discrete latent code.
    criterionQ_dis = nn.CrossEntropyLoss()
    # Loss for continuous latent code.
    criterionQ_con = NormalNLLLoss()

    # Adam optimiser is used.
    optimD = optim.Adam([{'params': discriminator.parameters()}, {'params': netD.parameters()}], lr=params.D.lr,
                        betas=(params.D.beta1, params.D.beta2))
    optimG = optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}], lr=params.G.lr,
                        betas=(params.G.beta1, params.G.beta2))
    # --------------------------------------------------------------------------

    real_label = 1
    fake_label = 0

    # List variables to store results pf training.
    img_list = defaultdict(list)

    print("-" * 25)
    print("Starting Training Loop...\n")
    print('Epochs: %d\nDataset: {}\nBatch Size: %d\nLength of Data Loader: %d'.format(params['dataset']) % (
        params['num_epochs'], params['batch_size'], len(dataloader)))
    print("-" * 25)

    for epoch in range(params['num_epochs']):
        print('Epoch:%03d' % epoch, '=' * 50)

        itqdm = tqdm(dataloader)
        for i, (data, _) in enumerate(itqdm):
            # Get batch size
            b_size = data.size(0)

            # Transfer data tensor to GPU/CPU (device), load from dataset
            real_data = data.to(device)

            # Updating discriminator and DHead
            optimD.zero_grad()
            # Real data, generate label corresponding to size of real_data
            label = torch.full((b_size,), real_label, device=device)
            output1 = discriminator(real_data)
            probs_real = netD(output1).view(-1)
            loss_real = criterionD(probs_real, label)
            # Calculate gradients.
            loss_real.backward()

            # Fake data, fill with value fake_label
            label.fill_(fake_label)
            noise, idx = noise_sample(params.num_dis_c, params.dis_c_dim, params.num_con_c, params.num_z, b_size,
                                      device)
            fake_data = netG(noise)
            output2 = discriminator(fake_data.detach())
            probs_fake = netD(output2).view(-1)
            loss_fake = criterionD(probs_fake, label)

            # Calculate gradients.
            loss_fake.backward()

            # Net Loss for the discriminator
            D_loss = loss_real + loss_fake
            # Update parameters
            optimD.step()

            # Updating Generator and QHead
            optimG.zero_grad()

            # Fake data treated as real.
            output = discriminator(fake_data)
            label.fill_(real_label)
            probs_fake = netD(output).view(-1)
            gen_loss = criterionD(probs_fake, label)

            q_logits, q_mu, q_var = netQ(output)
            target = torch.LongTensor(idx).to(device)

            # Calculating loss for discrete latent code.
            dis_loss = 0
            for j in range(params['num_dis_c']):
                dis_loss += criterionQ_dis(q_logits[:, j * params.dis_c_dim: j * params.dis_c_dim + params.dis_c_dim],
                                           target[j])

            # Calculating loss for continuous latent code.
            con_loss = 0
            if (params['num_con_c'] != 0):
                con_loss = criterionQ_con(
                    noise[:, params['num_z'] + params['num_dis_c'] * params['dis_c_dim']:].view(-1,
                                                                                                params['num_con_c']),
                    q_mu, q_var)

            # Net loss for generator.
            G_loss = gen_loss + params.coeff.dis * dis_loss + params.coeff.con * con_loss

            # Calculate gradients.
            G_loss.backward()
            # Update parameters.
            optimG.step()

            info = 'D %.4f， G %.4f' % (D_loss.item(), G_loss.item())

            itqdm.set_description(info)

            writer.add_scalar('G_loss/generator', gen_loss, epoch * len(dataloader) + i)
            writer.add_scalar('G_loss/discrete', dis_loss, epoch * len(dataloader) + i)
            writer.add_scalar('G_loss/continuous', con_loss, epoch * len(dataloader) + i)

            writer.add_scalar('D_loss/real', loss_real, epoch * len(dataloader) + i)
            writer.add_scalar('D_loss/fake', loss_fake, epoch * len(dataloader) + i)

            writer.add_scalar('Loss/D', D_loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar('Loss/G', G_loss.item(), epoch * len(dataloader) + i)

        # Generate image after each epoch to check performance of the generator. Used for creating animated gif later.
        with torch.no_grad():
            rows = 10
            fixed_noise = get_noise(params, device)
            var_noise = get_noiseByContinuous_code(params, device, rows=rows, columns=10)

            noises = {**fixed_noise, **var_noise}

            for key, noise in noises.items():
                gen_data = netG(noise).detach().cpu()

                images = vutils.make_grid(gen_data, nrow=rows, padding=2, normalize=True)

                if key.strip() == 'fixed':
                    writer.add_image('fix/%s' % key, images, epoch)
                else:
                    writer.add_image('train/%s' % key, images, epoch)

                img_list[key].append(images)

        # Save network weights.
        if (epoch + 1) % params['save_epoch'] == 0:
            torch.save(
                {'netG': netG.state_dict(), 'discriminator': discriminator.state_dict(), 'netD': netD.state_dict(),
                 'netQ': netQ.state_dict(), 'optimD': optimD.state_dict(), 'optimG': optimG.state_dict(),
                 'params': params}, osp.join(params['checkpoint'], 'model_epoch_%d.pth.tar' % (epoch + 1)))

    # Animation showing the improvements of the generator.
    show_improvement(img_list)


def show_improvement(img_lists):
    # Animation showing the improvements of the generator.
    for key, img_list in img_lists.items():
        fig = plt.figure(figsize=(10, 10))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
        anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        save_gif = osp.join(params['info'], '%s.gif' % key)
        anim.save(save_gif, dpi=80, writer='imagemagick')
        # plt.show()
        plt.close()

    img_list_path = osp.join(params['info'], 'img_list.npz')
    np.savez(img_list_path, img_lists=img_lists)


def main():
    output_folder = args.output_folder  # 'output1'
    params['summary'] = osp.join('./', output_folder, 'summary', params['dataset'] + args.comment)
    params['checkpoint'] = osp.join('./', output_folder, 'checkpoint', params['dataset'] + args.comment)
    params['info'] = osp.join('./', output_folder, 'info', params['dataset'] + args.comment)
    os.makedirs(params['summary'], exist_ok=True)
    os.makedirs(params['checkpoint'], exist_ok=True)
    os.makedirs(params['info'], exist_ok=True)

    writer = SummaryWriter(params['summary'])

    dataloader = get_data(params['dataset'], params['batch_size'])

    plot_training_image(dataloader, writer)

    train(dataloader, writer)
    writer.close()


if __name__ == '__main__':
    main()
