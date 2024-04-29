import os

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tensorboardX import SummaryWriter

from ProgGAN import ProgGAN


max_size = 256
data_dir = "celeba"
log_dir = "logs"
nimgs = 600000 # How many images to show the model before growing to the next size
checkpoint_every = 5000
log_every = 100
current_size = 4
current_step = 0
current_img = nimgs+1
epoch = 0
done = False
writer = SummaryWriter(log_dir=log_dir)

lr = 0.00002

checks = [4, 8, 16, 32, 64, 128, 256]
for i in checks:
    assert(os.path.isdir(os.path.join(data_dir, str(i))))

if not os.path.exists(log_dir):
    os.mkdir(log_dir)
if not os.path.exists(os.path.join(log_dir, "samples")):
    os.mkdir(os.path.join(log_dir, "samples"))

gan = ProgGAN()


def log(errD, errG, errD_real, errD_fake, errD_grad, errD_drift):
    print("Epoch %d Step %d // ErrD: %g ErrG: %g" % (epoch, current_step, float(errD), float(errG)))
    img = gan.sample(size=1, torch_style=True)[0]

    writer.add_image("fake", img, global_step=current_step)
    writer.add_scalar("errD", errD, global_step=current_step)
    writer.add_scalar("errG", errG, global_step=current_step)
    writer.add_scalar("errD_real", errD_real, global_step=current_step)
    writer.add_scalar("errD_fake", errD_fake, global_step=current_step)
    writer.add_scalar("errD_grad", errD_grad, global_step=current_step)
    writer.add_scalar("errD_drift", errD_drift, global_step=current_step)


def save():
    f = os.path.join(log_dir, "checkpoint_" + str(current_step) + ".th")
    state = {
        "GAN": gan.state_dict(),
        "current_step": current_step,
        "current_image": current_img,
        "epoch": epoch,
    }
    torch.save(state, f)


i = []
for root, dirs, files in os.walk(log_dir):
    for f in files:
        if f.split("_")[0] == "checkpoint":
            i.append(int(f.split("_")[-1].split(".")[0]))
if not len(i) == 0:
    i = sorted(i)[-1]
    f = os.path.join(log_dir, "checkpoint_" + str(i) + ".th")
    state = torch.load(f)
    gan.load(state["GAN"])
    current_size = gan.current_size
    current_step = state["current_step"]
    current_img = state["current_image"]
    epoch = state["epoch"]
    del state

batch_size = gan.batch_size
dataset = ImageFolder(root=os.path.join(data_dir, str(current_size)), transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
alpha_step = (1.0/(nimgs-1000))*batch_size
gan.set_lr(lr)
gan.cuda()

while not done:
    try:
        for data in dataloader:
            data = data[0]
            if data.size()[0] < batch_size:
                continue
            if current_img < nimgs:
                gan.increase_alpha(alpha_step)
            else:
                if gan.disc.alpha < 1.0:
                    print("Alpha: " + gan.disc.alpha)
                    gan.increase_alpha(alpha_step)

            if current_img > nimgs*2:
                if gan.current_size < max_size:
                    print("Growing GAN from size " + str(current_size) + " to " + str(current_size*2) + "!")
                    save()
                    gan.sample(path=os.path.join(log_dir, "samples", "sample_" + str(current_step) + ".png"))
                    gan.grow()
                    current_size *= 2
                    batch_size = gan.batch_size
                    dataset = ImageFolder(root=os.path.join(data_dir, str(current_size)), transform=transforms.Compose([
                        transforms.ToTensor(),
                     ]))
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                    alpha_step = (1.0/(nimgs-1000))*batch_size
                    current_img = 0
                    epoch = 0
                    break
                else:
                    print("Training done!")
                    save()
                    done = True
                    break

            errD, errG, errD_real, errD_fake, errD_grad, errD_drift = gan.train(data)

            if current_step % log_every == 0:
                log(errD, errG, errD_real, errD_fake, errD_grad, errD_drift)

            if current_step % checkpoint_every == 0:
                save()
                gan.sample(path=os.path.join(log_dir, "samples", "sample_" + str(current_step) + ".png"))

            current_step += 1
            current_img += batch_size

    except KeyboardInterrupt:
        print("Interrupted! Saving...")
        save()
        exit()
    epoch += 1

