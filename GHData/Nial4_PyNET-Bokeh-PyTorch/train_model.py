# model coding by Andrey Ignatov. recoded by Haiya Huang for Bokeh task

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from tqdm import tqdm
import torch
import imageio
import numpy as np
import math
import sys
from argparse import ArgumentParser
from dataset import LoadDataset
# from load_data import LoadData, LoadVisualData
from msssim import MSSSIM
from model_restr import PyNET
from vgg import vgg_19
from utils import normalize_batch, process_command_args
from time import strftime, localtime, time
from pytorch_msssim import SSIM, MS_SSIM
from visdom import Visdom

# import cv2
# from skimage.measure import compare_ssim as ssim


to_image = transforms.Compose([transforms.ToPILImage()])



np.random.seed(0)
torch.manual_seed(0)

# Processing command arguments
argv = ArgumentParser(usage='train parser', description='this is a parser')
argv.add_argument('--level', default=-3, type=int, help='the second argument')
argv.add_argument('--batch', default=50, type=int, help='the second argument')
argv.add_argument('--epoch', default=5, type=int, help='the second argument')
argv.add_argument('--restore_epoch', default=None,  help='the second argument')
args = argv.parse_args()
hello = "level:" + str(args.level) + "\nbatch size:" + str(args.batch) + "\nepoch:" + str(args.epoch)
print(hello)

level = args.level
batch_size = args.batch
learning_rate = 5e-5
restore_epoch = args.restore_epoch
num_train_epochs = args.epoch
# dslr_scale = float(1) / (2 ** (level - 1))

if level == 5:
    dslr_scale = 0.0625
if level == 4:
    dslr_scale = 0.125
if level == 3:
    dslr_scale = 0.25
if level == 2:
    dslr_scale = 0.5
if level == 1:
    dslr_scale = 1
if level == 0:
    dslr_scale = 2
if level == -1:
    dslr_scale = 4

# Dataset size

TRAIN_SIZE = 4000
TEST_SIZE = 694

# create log
log_path = "models/"
full_log_path = log_path + 'level' + str(level) + '.txt'
log_file = open(full_log_path, 'a+')
log_file.write(hello + '\n')
log_file.write(strftime("%Y-%m-%d %H:%M:%S", localtime()) + ' started \n')
log_file.write("==================\n")


def train_model():
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    print("CUDA visible devices: " + str(torch.cuda.device_count()))
    print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

    # Creating dataset loaders

    train_dataset = LoadDataset(root='/home/---------------------/train/',
                                mode='train', dslr_scale=dslr_scale, level=level)
    test_dataset = LoadDataset(root='/home/----------------------/test/',
                               mode='test', dslr_scale=dslr_scale, level=level)
    val_dataset = LoadDataset(root='/home/-----------------------/val/',
                              mode='val', dslr_scale=dslr_scale, level=level)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False
    )
    val_loaders = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=4,
        shuffle=False
    )


    # train_dataset = LoadData(dataset_dir, TRAIN_SIZE, dslr_scale, test=False)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
    #                           pin_memory=True, drop_last=True)
    #
    # test_dataset = LoadData(dataset_dir, TEST_SIZE, dslr_scale, test=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1,
    #                          pin_memory=True, drop_last=False)
    # visual_dataset = LoadVisualData(dataset_dir, 10, dslr_scale, level)
    # visual_loader = DataLoader(dataset=visual_dataset, batch_size=1, shuffle=False, num_workers=0,
    #                            pin_memory=True, drop_last=False)
    # Creating image processing network and optimizer
    generator = PyNET(level=level, instance_norm=True, instance_norm_level_1=True).to(device)
    generator = torch.nn.DataParallel(generator)

    # # Find total parameters and trainable parameters
    # total_params = sum(p.numel() for p in generator.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(
    #     p.numel() for p in generator.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')

    # optimizer = Adam(params=generator.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(
    #     generator.parameters(),
    #     lr=learning_rate,
    #     weight_decay=0.0001
    # )
    optimizer = torch.optim.Adam(params=generator.parameters(),
                                 lr=learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 weight_decay=0.0001)

    # Restoring the variables

    # if level < 3:
    #     generator.load_state_dict(torch.load("models/pynet_level_" + str(level + 1) +
    #                                          "_epoch_" + str(restore_epoch) + ".pth"), strict=False)

    # Losses

    VGG_19 = vgg_19(device)
    MSE_loss = torch.nn.MSELoss()
    L1_loss = torch.nn.L1Loss()
    # MS_SSIM = MSSSIM()
    SSIMX = SSIM(data_range=1, channel=3)


    viz = Visdom()

    viz.line([0.], [0], win='train_loss'+str(level), opts=dict(title='train_loss'+str(level)))
    viz.line([0.], [0], win='test_loss' + str(level), opts=dict(title='test_loss'+str(level)))
    # Train the network

    for epoch in range(num_train_epochs):

        torch.cuda.empty_cache()
        current_ep = epoch + 1

        loop = tqdm(
            train_loader,
            leave=True,
            desc=f"Train Epoch:{current_ep}/{num_train_epochs}"
        )

        train_iter = iter(train_loader)
        for i, (x, y) in enumerate(loop):
            optimizer.zero_grad()
            # x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            enhanced = generator(x)
            # MSE Loss
            loss_l1 = L1_loss(enhanced, y)

            # VGG Loss

            if level < 5:
                enhanced_vgg = VGG_19(normalize_batch(enhanced))
                target_vgg = VGG_19(normalize_batch(y))
                loss_content = MSE_loss(enhanced_vgg, target_vgg)

            # Total Loss

            if level == 5 or level == 4:
                total_loss = loss_l1 * 100
            if level == 3 or level == 2:
                total_loss = loss_l1 * 100 + loss_content * 10
            if level == 1:
                total_loss = loss_l1 * 100 + loss_content * 10
            if level == 0:
                loss_ssim = SSIMX(enhanced, y)
                total_loss = loss_l1 * 10 + loss_content * 0.1 + (1 - loss_ssim) * 10
            if level == -1:
                loss_ssim = SSIMX(enhanced, y)
                total_loss = loss_l1 * 10 + loss_content * 0.1 + (1 - loss_ssim) * 10


            # Perform the optimization step
            total_loss.backward()
            optimizer.step()

            if i == len(loop) - 1:  # len(loop) - 1
                # Save the model that corresponds to the current epoch
                generator.eval().cpu()
                if epoch % 5 == 0:
                    torch.save(generator.state_dict(),
                               "models/pynet_level_" + str(level) + "_epoch_" + str(epoch) + ".pth")
                    print("\n" + str(epoch) + " model saved")
                generator.to(device).train()

                # Save visual results for several test images

                generator.eval()
                with torch.no_grad():

                    visual_iter = iter(val_loaders)
                    for j in range(len(val_loaders)):
                        torch.cuda.empty_cache()

                        images = next(visual_iter)
                        x = images[0]
                        x = x.to(device, non_blocking=True)
                        enhanced = generator(x.detach())
                        enhanced = np.asarray(to_image(torch.squeeze(enhanced.detach().cpu())))

                        imageio.imwrite("results/pynet_img_" + str(j) + "_level_" + str(level) + "_epoch_" +
                                        str(epoch) + ".jpg", enhanced)

                # Evaluate the model
                print("start Test " + str(epoch) + "=======================================")
                loss_mse_eval = 0
                loss_l1_eval = 0
                loss_psnr_eval = 0
                loss_vgg_eval = 0
                loss_ssim_eval = 0

                generator.eval()
                with torch.no_grad():
                    test_iter = iter(test_loader)
                    for j in range(len(test_loader)):

                        x, y = next(test_iter)
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        enhanced = generator(x)

                        loss_mse_temp = MSE_loss(enhanced, y).item()
                        loss_l1_eval += L1_loss(enhanced, y).item()
                        loss_mse_eval += loss_mse_temp
                        loss_psnr_eval += 20 * math.log10(1.0 / math.sqrt(loss_mse_temp))

                        if level < 2:
                            loss_ssim_eval += SSIMX(y, enhanced)

                        if level < 5:
                            enhanced_vgg_eval = VGG_19(normalize_batch(enhanced)).detach()
                            target_vgg_eval = VGG_19(normalize_batch(y)).detach()
                            loss_vgg_eval += MSE_loss(enhanced_vgg_eval, target_vgg_eval).item()


                loss_mse_eval = loss_mse_eval / TEST_SIZE
                loss_psnr_eval = loss_psnr_eval / TEST_SIZE
                loss_vgg_eval = loss_vgg_eval / TEST_SIZE
                loss_ssim_eval = loss_ssim_eval / TEST_SIZE
                loss_l1_eval = loss_l1_eval / TEST_SIZE
                viz.line([loss_l1_eval * 100 + loss_vgg_eval * 10], [epoch], win='test_loss' + str(level),
                         update='append')

                if level < 2:
                    print("Evaluate Epoch %d, mse: %.4f, psnr: %.4f, vgg: %.4f, ms-ssim: %.4f,, L1: %.4f" % (epoch,
                                                                                                  loss_mse_eval,
                                                                                                  loss_psnr_eval,
                                                                                                  loss_vgg_eval,
                                                                                                  loss_ssim_eval,
                                                                                                  loss_l1_eval))
                elif level < 5:
                    print("Evaluate Epoch %d, mse: %.4f, psnr: %.4f, vgg: %.4f, L1: %.4f" % (epoch,
                                                                                   loss_mse_eval, loss_psnr_eval,
                                                                                   loss_vgg_eval,loss_l1_eval))
                else:
                    print("Evaluate Epoch %d, mse: %.4f, psnr: %.4f, L1: %.4f" % (epoch, loss_mse_eval, loss_psnr_eval,loss_l1_eval))

                print("End Test " + str(epoch) + "========================================")
                log_file.write("==================\n")
                log_file.write("Test epoch:" + str(epoch) + "\n" +
                               "loss_mse_eval: " + str(loss_mse_eval) + "\n" +
                               "loss_psnr_eval: " + str(loss_psnr_eval) + "\n" +
                               "loss_vgg_eval: " + str(loss_vgg_eval) + "\n" +
                               "loss_ssim_eval: " + str(loss_ssim_eval) + "\n" +
                               "loss_l1_eval: " + str(loss_l1_eval) + "\n" +
                               strftime("%Y-%m-%d %H:%M:%S", localtime()) +
                               '\n')
                log_file.write("==================\n")

                generator.train()

                loop.set_postfix(
                    lr=optimizer.param_groups[0]['lr'],
                    loss=total_loss.item(),
                    # content=loss_content,
                    # ssim=loss_ssim,
                    # content=me_loss_content,
                    # mse=me_loss_mse
                )

        viz.line([total_loss.item()], [epoch], win='train_loss'+str(level), update='append')


    torch.save(generator.state_dict(), "models/pynet_level_" + str(level) + "_epoch_" + "None" + ".pth")


if __name__ == '__main__':
    train_model()
    log_file.close()
    print("end")
