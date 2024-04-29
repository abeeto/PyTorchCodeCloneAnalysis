import torch
import torch.nn as nn
import torch.optim as optim
import config
import os
from dataset import TrainDataset, TestDataset
from generator_model import Generator
from discriminator_model import Discriminator
from VGGNet import VGGNet
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from utils import save_test_examples, load_checkpoint, save_checkpoint, save_training_images

def initialization_phase(gen, loader, opt_gen, l1_loss, VGG, pretrain_epochs):
    for epoch in range(pretrain_epochs):
        loop = tqdm(loader, leave=True)
        losses = []

        for idx, (sample_photo, _, _) in enumerate(loop):
            sample_photo = sample_photo.to(config.DEVICE)

            # train generator G
            #with torch.cuda.amp.autocast():
            reconstructed = gen(sample_photo)

            sample_photo_feature = VGG(sample_photo)
            reconstructed_feature = VGG(reconstructed)
            reconstruction_loss = l1_loss(reconstructed_feature, sample_photo_feature.detach())
            
            losses.append(reconstruction_loss.item())

            opt_gen.zero_grad()
            
            reconstruction_loss.backward()
            opt_gen.step()

            #opt_gen.zero_grad()
            #g_scaler.scale(reconstruction_loss).backward()
            #g_scaler.step(opt_gen)
            #g_scaler.update()    
            loop.set_postfix(epoch=epoch + 1)

        print('[%d/%d] - Recon loss: %.8f' % ((epoch + 1), pretrain_epochs, torch.mean(torch.FloatTensor(losses))))
        
        save_image(sample_photo*0.5+0.5, os.path.join(config.RESULT_TRAIN_DIR, str(epoch + 1) + "_initialization_phase_photo.png"))
        save_image(reconstructed*0.5+0.5, os.path.join(config.RESULT_TRAIN_DIR, str(epoch + 1) + "_initialization_phase_reconstructed.png"))


def train_fn(total_epoch, disc, gen, train_loader, val_loader, opt_disc, opt_gen, l1_loss, mse, VGG):
    step = 0 

    for epoch in range(total_epoch):
        loop = tqdm(train_loader, leave=True)

        # Training
        for idx, (sample_photo, sample_cartoon, sample_edge) in enumerate(loop):
            sample_photo = sample_photo.to(config.DEVICE)
            sample_cartoon = sample_cartoon.to(config.DEVICE)
            sample_edge = sample_edge.to(config.DEVICE)

            # Train Discriminator
            #Pass samples into Discriminator: Fake Cartoon, real Cartoon and Edge
            fake_cartoon = gen(sample_photo)

            D_real = disc(sample_cartoon)
            D_fake = disc(fake_cartoon.detach())
            D_edge = disc(sample_edge)

            #Compute loss (edge-promoting adversarial loss)
            D_real_loss = mse(D_real, torch.ones_like(D_real))
            D_fake_loss = mse(D_fake, torch.zeros_like(D_fake))
            D_edge_loss = mse(D_edge, torch.zeros_like(D_edge))

            # Author's code divided it by 3.0, I believe it has similar thoughts to CycleGAN (divided by 2 with only 2 loss)
            D_loss = (D_real_loss + D_fake_loss + D_edge_loss) / 3.0   
                
            opt_disc.zero_grad() # clears old gradients from the last step

            D_loss.backward()
            opt_disc.step()

            # Train Generator
            D_real = disc(sample_cartoon)
            D_fake = disc(fake_cartoon.detach())

            G_fake_loss = mse(D_fake, torch.ones_like(D_fake))

            # Content loss
            sample_photo_feature = VGG(sample_photo)
            fake_cartoon_feature = VGG(fake_cartoon)
            content_loss = l1_loss(fake_cartoon_feature, sample_photo_feature.detach())

            # Compute loss (adversarial loss + lambda*content loss)
            G_loss = config.LAMBDA_ADV * G_fake_loss + config.LAMBDA_CONTENT * content_loss

            opt_gen.zero_grad()

            G_loss.backward()
            opt_gen.step()

            if step % config.SAVE_IMG_PER_STEP == 0:
                save_image(sample_photo*0.5+0.5, os.path.join(config.RESULT_TRAIN_DIR, "epoch_" + str(epoch + 1) + "step_" + str(step + 1) + "_photo.png"))
                save_image(fake_cartoon*0.5+0.5, os.path.join(config.RESULT_TRAIN_DIR, "epoch_" + str(epoch + 1) + "step_" + str(step + 1) + "_fakecartoon.png"))
                print('[Epoch: %d| Step: %d] - D loss: %.12f' % ((epoch + 1), (step+1), D_loss.item()))
                print('[Epoch: %d| Step: %d] - G Content loss (w/o lambda): %.12f' % ((epoch + 1), (step+1), content_loss.item()))
                print('[Epoch: %d| Step: %d] - G Content loss (w/ lambda): %.12f' % ((epoch + 1), (step+1), config.LAMBDA_CONTENT * content_loss.item()))
                print('[Epoch: %d| Step: %d] - G ADV loss (w/o lambda): %.12f' % ((epoch + 1), (step+1), G_fake_loss.item()))
                print('[Epoch: %d| Step: %d] - G ADV loss (w/ lambda): %.12f' % ((epoch + 1), (step+1), config.LAMBDA_ADV * G_fake_loss.item()))
                print('[Epoch: %d| Step: %d] - G loss: %.12f' % ((epoch + 1), (step+1), G_loss.item()))

            step+= 1

            loop.set_postfix(step=step, epoch=epoch+1)

        if config.SAVE_MODEL and epoch % config.SAVE_MODEL_FREQ == 0:
            save_checkpoint(gen, opt_gen, epoch, folder=config.CHECKPOINT_FOLDER, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, epoch, folder=config.CHECKPOINT_FOLDER, filename=config.CHECKPOINT_DISC)

        # Test Some data
        save_test_examples(gen, val_loader, epoch, folder=config.RESULT_TEST_DIR)


def main():
    print(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    VGG19 = VGGNet(in_channels=3, VGGtype="VGG19", init_weights=config.VGG_WEIGHTS, batch_norm=False, feature_mode=True)
    VGG19 = VGG19.to(config.DEVICE)
    VGG19.eval()

    if config.LOAD_MODEL:
        is_gen_loaded = load_checkpoint(
            gen, opt_gen, config.LEARNING_RATE, folder=config.CHECKPOINT_FOLDER, checkpoint_file=config.LOAD_CHECKPOINT_GEN
        )
        is_disc_loaded = load_checkpoint(
            disc, opt_disc, config.LEARNING_RATE, folder=config.CHECKPOINT_FOLDER, checkpoint_file=config.LOAD_CHECKPOINT_DISC
        )
    
    #BCE_Loss = nn.BCELoss()
    L1_Loss = nn.L1Loss()
    MSE_Loss = nn.MSELoss() # went through the author's code and found him using LSGAN, LSGAN should gives better training

    
    train_dataset = TrainDataset(config.TRAIN_PHOTO_DIR, config.TRAIN_CARTOON_EDGE_DIR)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    val_dataset = TestDataset(config.VAL_PHOTO_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)

    # Initialization Phase
    if not(is_gen_loaded):
        print("="*80)
        print("=> Initialization Phase")
        initialization_phase(gen, train_loader, opt_gen, L1_Loss, VGG19, pretrain_epochs=config.PRETRAIN_EPOCHS)
        print("Finished Initialization Phase")
        print("="*80)

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, 'i', folder=config.CHECKPOINT_FOLDER, filename=config.CHECKPOINT_GEN)

    # Do the training
    print("=> Start Training")
    train_fn(config.NUM_EPOCHS, disc, gen, train_loader, val_loader, opt_disc, opt_gen, L1_Loss, MSE_Loss, VGG19)
    print("Finished Training")

if __name__ == "__main__":
    main()

#TODO
#https://github.com/SystemErrorWang/CartoonGAN/blob/master/old_code/main.py