from model.generator import Generator
from model.discriminator import Discriminator
from utils.function import prepare_data, ritagliare_centro, create_dir, create_graphic_testing
from utils.parameters import *

import torchvision.utils as vutils

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

create_dir(TEST_RESULT)

criterion = nn.BCELoss()

generator = Generator()
generator.load_state_dict(torch.load("./log/generator.pt"))
generator.eval()

discriminator = Discriminator()
discriminator.load_state_dict(torch.load("./log/discriminator.pt"))
discriminator.eval()

dataloader = prepare_data("./dataset/testing/")

num_img = 0
perdita = 0.0
total = 0.0

for data in dataloader:
    real_cpu, _ = data
    real_center_cpu = real_cpu[:, :, int(img_size / 4):int(img_size / 4) + int(img_size / 2),
                      int(img_size / 4):int(img_size / 4) + int(img_size / 2)]
    batch_size = real_cpu.size(0)
    real_cpu = real_cpu.cuda()
    real_center_cpu = real_center_cpu.cuda()
    real_cpu.to(device)
    real_center_cpu.to(device)

    # Individuiamo e ritagliamo il centro dell'immagine reale
    input_real, input_cropped, real_center = ritagliare_centro(input_real, input_cropped, real_cpu, real_center,
                                                               real_center_cpu)

    with torch.no_grad():
        label.resize_((batch_size, 1, 1, 1)).fill_(real_label)

    fake = generator(input_cropped)
    label.data.fill_(fake_label)
    output = discriminator(fake.detach())
    errD_fake = criterion(output, label)
    total += 1
    perdita += errD_fake.mean().item()

    ricostruzione.append(100 * perdita / total)

    recon_image = input_cropped.clone()
    recon_image.data[:, :, int(img_size / 4):int(img_size / 4 + img_size / 2),
                        int(img_size / 4):int(img_size / 4 + img_size / 2)] = fake.data


    for i in range(0, batch_size):
        num_img += 1
        vutils.save_image([input_real[i], input_cropped[i], recon_image[i]],
                          TEST_RESULT + f"ricostruite_{num_img}.png")

print(f"Accuratezza: {100 * perdita / total}")
create_graphic_testing(ricostruzione)
