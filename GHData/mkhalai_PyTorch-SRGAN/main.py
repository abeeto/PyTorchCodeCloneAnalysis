from config import *
from dataset import TrainDataset
from loss import adversarial_loss, vgg_loss, discriminator_loss, \
    mseLoss, discriminator_accuracy
from model import GeneratorNetwork, DiscriminatorNetwork
import os
from torch import optim
from torch.utils.data import DataLoader
import utils


def train(generator, discriminator, genOpt, discOpt, dataset, num_epochs = NUM_EPOCHS):
    generator.train()
    discriminator.train()
    
    for epoch in range(1, num_epochs+1):
        
        for batch, (lr,hr) in enumerate(dataset):
            lr = lr.to(device)
            hr = hr.to(device)

            sr = generator(lr)
            
            disc_hr = discriminator(hr)
            disc_sr = discriminator(sr.detach())
            disc_loss = discriminator_loss(disc_hr, disc_sr)
            disc_acc = discriminator_accuracy(disc_hr, disc_sr)

            discOpt.zero_grad()
            disc_loss.backward()
            discOpt.step()
            
            #--------Generator training------------
            disc_sr = discriminator(sr)
            adv_loss = adversarial_loss(disc_sr)
            content_loss = vgg_loss(sr,hr)
            
            mse_loss = mseLoss(hr,sr)
            #generator_loss = 0.006*content_loss + 1e-4*adv_loss
            generator_loss =  mse_loss
            
            genOpt.zero_grad()
            generator_loss.backward()
            genOpt.step()
            

            print("epoch: %2d, batch: [%2d/%2d], Generator loss: %.6f" % (epoch, batch+1, NUM_BATCHES, generator_loss.item()))
            #print("epoch: %2d, batch: [%2d/%2d], Disc loss: %.6f, Disc acc: %.3f" % (epoch, batch+1, NUM_BATCHES, disc_loss.item(), disc_acc))
            
            #print("epoch: %2d, batch: [%2d/%2d], Generator loss: %.6f, Discriminator loss: %.6f" 
             #       % (epoch,batch+1,NUM_BATCHES,generator_loss.item(), disc_loss.item()))
            
            if batch % 5 == 0: 
                utils.save_image(generator=generator, epoch=epoch)
                
        if (epoch % 1 == 0):
            print('=> Saving models..')
            
            utils.save_checkpoint(generator, genOpt, GEN_PATH)
            utils.save_checkpoint(discriminator, discOpt, DISC_PATH)
            print('=> Checkpoint saved.')


def main():
    generator = GeneratorNetwork().to(device)
    discriminator = DiscriminatorNetwork().to(device)
    generatorOptim = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    discriminatorOptim = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    
    trainDataset = TrainDataset(IMAGE_PATH)
    trainDataLoader = DataLoader(trainDataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True
    )
    print('check1')
    if os.path.exists(GEN_PATH):
        print('=> Loading models..')
        utils.load_checkpoint(generator, generatorOptim, GEN_PATH, 0.0001)
        utils.load_checkpoint(discriminator, discriminatorOptim, DISC_PATH, 0.0001)
        print('=> Loading complete.')
    print('check2')

    train(generator, discriminator, genOpt=generatorOptim, discOpt=discriminatorOptim, dataset=trainDataLoader)

if __name__ == '__main__':
    main()

