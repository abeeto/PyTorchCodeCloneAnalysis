#import sys
#sys.path.append('model')

from PIL import Image
import datetime
from pathlib import Path
import random
import tqdm
import torch.optim as optim
import torch.utils.tensorboard as tensorboard

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from weights_init import weights_init
from init_train import init_train
from model.generator import Generator
from model.discriminator import Discriminator
from dataset import Dataset
from loss import VGGLosses


def train(epoch : int = 10,
          adv_weight : float = 1.0,
          threshold : float = 0.,
          G_train_iter : int = 1,
          D_train_iter : int = 1):    #if threshold is 0., set to half of adversarial loss

  test_img_dir = Path('./dataset/test/test_photo256').resolve()
  test_img_dir = random.choice(list(test_img_dir.glob('**/*')))
  test_img = Image.open(test_img_dir)
  current_time = datetime.datetime.now().strftime("%H:%M:%S")
  writer.add_image('test image {}'.format(current_time), np.asarray(test_img), dataformats='HWC')
  writer.flush()
  perception_weight = 0.
  keep_constant = False


  for epoch in tqdm(range(epoch)):
      total_dis_loss = 0.

      for i, (style, smooth, train) in enumerate(dataloader, 0):

        D.zero_grad()

        train = train.to(device)
        style = style.to(device)
        #smooth = smooth.to(device)

        for _ in range(D_train_iter):
          style_loss_value = D(style).view(-1)
          generator_output = G(train)
          real_output = D(generator_output.detach()).view(-1)
          dis_adv_loss = adv_weight * (torch.pow(style_loss_value - 1, 2).mean() + torch.pow(real_output, 2).mean())
          total_dis_loss += dis_adv_loss.item()
          dis_adv_loss.backward()
        optimizer_D.step()

        G.zero_grad()
        for _ in range(G_train_iter):
          generator_output = G(train)
          real_output = D(generator_output).view(-1)
          per_loss = perception_weight * vggloss.perceptual_loss(train, generator_output)
          gen_adv_loss = adv_weight * torch.pow(real_output - 1, 2).mean()
          gen_loss = gen_adv_loss + per_loss
          gen_loss.backward()
        optimizer_G.step()

        if i % 200 == 0 and i != 0:
            writer.add_scalars('generator losses {}'.format(current_time),
                              {'adversarial loss': dis_adv_loss.item(),
                                'Generator adversarial loss': gen_adv_loss.item(),
                                'perceptual loss' : per_loss.item()}, i + epoch * len(dataloader))
            writer.flush()

      if total_dis_loss > threshold and not keep_constant:
        perception_weight += 0.05
      else:
        keep_constant = True

      writer.add_scalar('total discriminator loss {}'.format(current_time), total_dis_loss, i + epoch * len(dataloader))


      G.eval()

      styled_test_img = transform(test_img).unsqueeze(0).to(device)
      with torch.no_grad():
        styled_test_img = G(styled_test_img)

      styled_test_img = styled_test_img.to('cpu').squeeze()
      write_image(writer, styled_test_img, 'styled image {}'.format(current_time), epoch + 1)

      G.train()

def main():

    if __name__ == '__main__':
        writer = tensorboard.SummaryWriter(log_dir = './logs')
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        vggloss = VGGLosses(device = device).to(device)

        dataset = Dataset(root = 'dataset/Shinkai',
                          style_transform = transform,
                          smooth_transform = transform)

        dataloader = DataLoader(dataset,
                                batch_size = 16,
                                shuffle = True)

        G = Generator().to(device)
        D = PatchDiscriminator().to(device)

        G.apply(weights_init)
        D.apply(weights_init)

        optimizer_G = optim.Adam(G.parameters(), lr = 0.0001)   #Based on paper
        optimizer_D = optim.Adam(D.parameters(), lr = 0.0004)  #Based on paper

        init_train(20, lr = 0.1, con_weight = 1.0)
        train(epoch = 10, con_weight = 1.2, gra_weight = 2., col_weight = 10.)
