import logging
import torch
import torch.nn as nn
import torch.optim as optim
import config
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, plot_examples , plot_reals
from model import Discriminator, Generator, initialize_weights

writer = SummaryWriter()
transforms = transforms.Compose(
    [
        transforms.Resize((config.IMAGE_SIZE,config.IMAGE_SIZE)) ,
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(config.CHANNELS_IMG)] , [0.5 for _ in range(config.CHANNELS_IMG)]),
    ]
    )

def train_fn(loader , critic , gen , opt_gen , opt_critic ,  fixed_noise, epoch , criterion):
    
    for batch_idx , (real , _) in enumerate(loader):
            real = real.to(config.device)
            cur_batch_size = real.shape[0]
            noise = torch.randn(cur_batch_size , config.Z_DIM , 1 , 1).to(config.device)
            fake = gen(noise)
            disc_real = critic(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = critic(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            critic.zero_grad()
            loss_disc.backward()
            opt_critic.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            output = critic(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            logging.info(
                    f"Epoch [{epoch}/{config.NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
                )
            
            writer.add_scalar("D_Loss/Losses", loss_disc , epoch)
            writer.add_scalar("G_Loss/Losses", loss_gen , epoch)
            

            if batch_idx % 61 == 0:
                with torch.no_grad():
                    fake = gen(fixed_noise)

                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize = True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize = True)
                    plot_examples( "Fake_ep"+str(epoch) + ".png" ,"TestFolder/" , img_grid_fake)
                    #plot_reals("Real_ep"+str(epoch)+ ".png" , "TestFolder/" , img_grid_real)
    



def main():

    dataset = datasets.ImageFolder(root=config.ROOT_DIR, transform=transforms)
    loader = DataLoader(
        dataset,
        batch_size = config.BATCH_SIZE,
        shuffle=True,
    )

    gen = Generator(config.Z_DIM , config.CHANNELS_IMG , config.FEATURES_GEN).to(config.device)
    critic  = Discriminator(config.CHANNELS_IMG , config.FEATURES_CRITIC).to(config.device)



    opt_gen = optim.Adam(gen.parameters() , lr = config.G_LEARNING_RATE , betas = (0.5 , 0.999))
    opt_critic = optim.Adam(critic.parameters() , lr = config.D_LEARNING_RATE , betas = (0.5 , 0.999))
    criterion = nn.BCELoss()


    if config.LOAD_MODEL:
        load_checkpoint(
                config.CHECKPOINT_GEN, gen, opt_gen, config.G_LEARNING_RATE,
            )
        load_checkpoint(
                config.CHECKPOINT_DISC , critic , opt_critic , config.D_LEARNING_RATE,
            )
    else:
        initialize_weights(gen)
        initialize_weights(critic)

    fixed_noise = torch.randn(32 , config.Z_DIM , 1 , 1).to(config.device)

#writer_real = SummaryWriter(f"logs/real")

    step = 0


    gen.train()
    critic.train()

    print("Train Starts")

    for epoch in range(config.NUM_EPOCHS):

        train_fn(loader , critic , gen , opt_gen , opt_critic , fixed_noise, epoch , criterion)

        if config.SAVE_MODEL:
            save_checkpoint(gen , opt_gen , filename = config.CHECKPOINT_GEN)
            save_checkpoint(critic , opt_critic , filename = config.CHECKPOINT_DISC)

        step += 1 

        
if __name__  == "__main__":
    logging.basicConfig(level = logging.DEBUG)
    torch.backends.cudnn.benchmark = True
    main()
