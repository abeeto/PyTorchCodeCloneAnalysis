import hydra
from omegaconf import DictConfig
from dataset import find_dataset_ref
from torch.utils.data import DataLoader
from model import RenderModel, Discriminator
from loss import VGGPerceptualLoss
import torch
from ema_pytorch import EMA
import wandb
import sys
import warnings
from PIL import Image
from utils import tensor_to_device, Notify, setup_seed
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio


warnings.filterwarnings("ignore")
isDebug = True if sys.gettrace() else False
device = 'cuda'


@hydra.main(config_path="config", config_name="cfg_spaces")
def main(cfg: DictConfig):
    # set seed
    setup_seed(0)

    # set use_wandb or not
    use_wandb = cfg.use_wandb and (not isDebug)
    if use_wandb:
        wandb.init(project="rendering", name=cfg.exp_name)

    # dataloader
    NVSDataset = find_dataset_ref(cfg.dataset_name)
    train_dataset = NVSDataset(cfg.data_folder,
                               'training',
                               cfg.n_views,
                               cfg.n_depths,
                               cfg.max_h,
                               cfg.max_w)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True,
                              drop_last=True)

    # model
    model_g = RenderModel().to(device)
    model_d = Discriminator().to(device)

    # loss definement
    loss_vgg = VGGPerceptualLoss(device=device)
    
    # optimizer
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))

    # TODO: ema
    
    # set model train
    model_g.train()
    model_d.train()

    # run epochs
    for epoch in tqdm(range(cfg.epochs)):
        for i, sample in enumerate(train_loader):
            sample = tensor_to_device(sample, device)
            src_imgs_BN3HW = sample['src_imgs']
            src_cams_BN244 = sample['src_cams']
            tgt_img_B3HW = sample['tgt_img']
            tgt_cam_B244 = sample['tgt_cam']
            depths_BD = sample['depths']

            # log input images
            src_imgs_np = (src_imgs_BN3HW.detach().cpu().numpy()[0] + 1) / 2
            tgt_img_np = (tgt_img_B3HW.detach().cpu().numpy()[0] + 1) / 2
            src_imgs_np *= 255
            tgt_img_np *= 255
            src_imgs_np = src_imgs_np.astype('uint8').transpose(0, 2, 3, 1)
            tgt_img_np = tgt_img_np.astype('uint8').transpose(1, 2, 0)
            tgt_img_pil = Image.fromarray(tgt_img_np)
            src_imgs_pil = [Image.fromarray(img) for img in src_imgs_np]

            # train generator
            aggregated_img_B3HW, final_out_img_B3HW, warped_imgs_BN3HW = \
                model_g(src_imgs_BN3HW, src_cams_BN244, tgt_cam_B244, depths_BD)
            predict_fake = model_d(final_out_img_B3HW)
            loss_gen_GAN = torch.mean(-torch.log(predict_fake + 1e-12))
            loss_gen_vgg = loss_vgg(final_out_img_B3HW, tgt_img_B3HW)
            loss_gen = loss_gen_GAN + loss_gen_vgg
            optimizer_g.zero_grad()
            loss_gen.backward()
            optimizer_g.step()

            # train discriminator
            predict_real = model_d(tgt_img_B3HW)
            predict_fake = model_d(final_out_img_B3HW.detach())
            loss_discrim = 0.5 * torch.mean(-(torch.log(predict_real + 1e-12) + torch.log(1 - predict_fake + 1e-12)))
            loss_gen_L1 = torch.mean(torch.abs(final_out_img_B3HW - tgt_img_B3HW))
            optimizer_d.zero_grad()
            loss_discrim.backward()
            optimizer_d.step()

            # log output images
            aggregated_img_np = (aggregated_img_B3HW.detach().cpu().numpy()[0] + 1) / 2
            final_out_img_np = (final_out_img_B3HW.detach().cpu().numpy()[0] + 1) / 2
            aggregated_img_np *= 255
            final_out_img_np *= 255
            aggregated_img_np = aggregated_img_np.astype('uint8').transpose(1, 2, 0)
            final_out_img_np = final_out_img_np.astype('uint8').transpose(1, 2, 0)
            aggregated_img_pil = Image.fromarray(aggregated_img_np)
            final_out_img_pil = Image.fromarray(final_out_img_np)

            # psnr
            psnr = peak_signal_noise_ratio(tgt_img_np, final_out_img_np)

            # print loss info
            if i % 20 == 0:
                print(Notify.INFO, f'epoch: {epoch}, iter: {i}, loss_discrim: {loss_discrim}, loss_gen: {loss_gen}, psnr: {psnr}', Notify.ENDC)
                if use_wandb:
                    wandb.log({'tgt_img': [wandb.Image(tgt_img_pil)]})
                    wandb.log({'src_imgs': [wandb.Image(img) for img in src_imgs_pil]})
                    wandb.log({'aggregated_img': [wandb.Image(aggregated_img_pil)]})
                    wandb.log({'final_out_img': [wandb.Image(final_out_img_pil)]})

            if use_wandb:
                wandb.log({
                    'train loss_discrim': loss_discrim.item(),
                    'train loss_generator': loss_gen.item(),
                    'train psnr': psnr 
                })
    
    # save models
    torch.save(model_d.state_dict(), 'd.pth')
    torch.save(model_g.state_dict(), 'g.pth')


if __name__ == "__main__":
    main()