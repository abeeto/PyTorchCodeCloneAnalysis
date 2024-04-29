import argparse
from dataset import load_dataset
from nets import *
from losses import *
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='./dataset/spade_celebA', help='The path of dataset')

    parser.add_argument('--img_height', type=int, default=256, help='The height size of image')
    parser.add_argument('--img_width', type=int, default=256, help='The width size of image ')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--segmap_ch', type=int, default=3, help='The size of segmap channel')
    parser.add_argument('--augment_flag', type=bool, default=False, help='Image augmentation use or not')

    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size') # batch size를 1에서 16으로 늘렸더니 gpu memory leak가 나타났다.
    parser.add_argument('--random_style', type=bool, default=False, help='if randmo style is false, it means "use an encoder"')
    
    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')

    parser.add_argument('--n_scale', type=int, default=2, help='number of scales')
    parser.add_argument('--n_dis', type=int, default=4, help='The number of discriminator layer')

    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')

    parser.add_argument('--adv_weight', type=int, default=1, help='Weight about GAN')
    parser.add_argument('--vgg_weight', type=int, default=10, help='Weight about perceptual loss')
    parser.add_argument('--feature_weight', type=int, default=10, help='Weight about discriminator feature matching loss')
    parser.add_argument('--kl_weight', type=float, default=0.05, help='Weight about kl-divergence')
    parser.add_argument('--gan_type', type=str, default='gan', help='gan / lsgan / hinge / wgan-gp / wgan-lp / dragan')

    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate')

    parser.add_argument('--decay_flag', type=bool, default=True, help='The decay_flag')
    parser.add_argument('--decay_epoch', type=int, default=50, help='decay epoch')
    parser.add_argument('--n_critic', type=int, default=1, help='The number of critic')

    parser.add_argument('--label_nc', type=int, default=19, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.') # 이것도 나중에 넘겨주는 형식으로 바꿀 수 있을 것 같다. -> 가장 나중에 하기

    parser.add_argument('--display_step', type=int, default=3000, help='size of results is display_step * display_step')

    parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
    parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')
    parser.add_argument('--use_vae', action='store_true', help='enable training with an image encoder.')

    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--crop_size', type=int, default=512, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')

    parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')

    parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
    parser.add_argument('--contain_dontcare_label', action='store_true', help='if the label map contains dontcare label (dontcare=255)')

    parser.add_argument('--norm_G', type=str, default='spectralspadeinstance3x3', help='instance normalization or batch normalization')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

    parser.add_argument('--netD_subarch', type=str, default='n_layer', help='architecture of each discriminator')
    parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to be used in multiscale')
    parser.add_argument('--n_layers_D', type=int, default=4, help='# layers in each discriminator')

    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
    parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

    parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
    parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')

    return parser.parse_args()

def imsave(input, name): # batch_size=1일 때
    input = input.detach()
    input = torch.squeeze(input) # batch 차원을 없애줬다.
    input = input.cpu().numpy().transpose((1, 2, 0))
    plt.imshow((input * 255).astype(np.uint8))
    plt.savefig(name)

def discriminate(netD, input_semantics, fake_image, real_image):
        # print(input_semantics.shape, fake_image.shape, real_image.shape)
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = netD(fake_and_real)

        pred_fake, pred_real = divide_pred(discriminator_out)

        return pred_fake, pred_real

# Take the prediction of fake and real images from the combined batch
def divide_pred(pred):
    # the prediction contains the intermediate outputs of multiscale GAN,
    # so it's usually a list
    if type(pred) == list:
        fake = []
        real = []
        for p in pred:
            fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
            real.append([tensor[tensor.size(0) // 2:] for tensor in p])
    else:
        fake = pred[:pred.size(0) // 2]
        real = pred[pred.size(0) // 2:]

    return fake, real

def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std) + mu

if __name__ == "__main__":
    args = parse_args()

    train_dataloader = load_dataset(args)

    netG, netD, netE = Generator(args).to(device).train(), MultiscaleDiscriminator(args).to(device).train(), None
    if args.use_vae:
        netE = Encoder(args).to(device).train()

    criterionGAN = GANLoss(opt=args)
    criterionFeat = torch.nn.L1Loss()
    criterionVGG = VGGLoss()
    KLDLoss = KLDLoss()

    G_params = list(netG.parameters())
    if args.use_vae:
        G_params += list(netE.parameters())
    D_params = list(netD.parameters())

    beta1, beta2 = args.beta1, args.beta2
    if args.no_TTUR:
        G_lr, D_lr = args.lr, args.lr
    else:
        G_lr, D_lr = args.lr / 2, args.lr * 2

    optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

    start_time = time.time()
    old_lr = args.lr

    for epoch in range(args.epoch):
        for idx, (real_x, label) in tqdm(enumerate(train_dataloader)):
            # print(f"idx is {idx}")
            label = label.long()
            real_x, label = real_x.to(device), label.to(device)
            # print(label.shape)
            
            # create one-hot label map
            label_map = label
            bs, _, h, w = label_map.size()
            nc = args.label_nc + 1 if args.contain_dontcare_label else args.label_nc
            FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
            input_label = FloatTensor(bs, nc, h, w).zero_()
            input_semantics = input_label.scatter_(1, label_map, 1.0)

            # train generator
            if idx % args.n_critic == 0:
                
                optimizer_G.zero_grad()
                
                z = None
                ## kld loss
                KLD_loss = torch.FloatTensor([0]).to(device)
                if args.use_vae:
                    mu, logvar = netE(real_x)
                    z = reparameterize(mu, logvar)
                    KLD_loss = KLDLoss(mu, logvar) * args.kl_weight
                KLD_loss.requires_grad_(True)

                fake_x = netG(input_semantics, z=z)
                # 강제로 fake_x resize...
                tf = transforms.Resize((args.img_height, args.img_width), interpolation=Image.BICUBIC)
                fake_x = tf(fake_x)
                pred_fake, pred_real = discriminate(netD, input_semantics, fake_x, real_x)

                ## gan loss
                loss = []
                for i in range(len(pred_fake)):
                    gan_loss = torch.mean(F.binary_cross_entropy_with_logits(input=pred_fake[i][-1], target=torch.ones_like(pred_fake[i][-1])))    
                    loss.append(gan_loss)
                GAN_loss = torch.mean(torch.FloatTensor(loss)).to(device)
                GAN_loss.requires_grad_(True)
                
                ## feat loss
                num_D = len(pred_fake)
                GAN_Feat_loss = FloatTensor(1).fill_(0)
                for i in range(num_D):  # for each discriminator
                    # last output is the final prediction, so we exclude it
                    num_intermediate_outputs = len(pred_fake[i]) - 1
                    for j in range(num_intermediate_outputs):  # for each layer output
                        unweighted_loss = criterionFeat(
                            pred_fake[i][j], pred_real[i][j].detach())
                        GAN_Feat_loss += unweighted_loss * args.feature_weight / num_D
                
                ## vgg loss
                VGG_loss = criterionVGG(fake_x, real_x) * args.vgg_weight
                
                g_loss = (KLD_loss + GAN_loss + GAN_Feat_loss + VGG_loss) / 4
                g_loss.backward()
                optimizer_G.step()
            
            # train discriminator
            optimizer_D.zero_grad()

            loss = []
            for i in range(len(pred_fake)):
                gan_loss = torch.mean(F.binary_cross_entropy_with_logits(input=pred_fake[i][-1], target=torch.zeros_like(pred_fake[i][-1])))    
                loss.append(gan_loss)
            FAKE_loss = torch.mean(torch.FloatTensor(loss)).to(device)
            FAKE_loss.requires_grad_(True)

            loss = []
            for i in range(len(pred_real)):
                gan_loss = torch.mean(F.binary_cross_entropy_with_logits(input=pred_real[i][-1], target=torch.zeros_like(pred_real[i][-1])))    
                loss.append(gan_loss)
            REAL_loss = torch.mean(torch.FloatTensor(loss)).to(device)
            REAL_loss.requires_grad_(True)

            d_loss = (FAKE_loss + REAL_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            if idx % args.display_step == 0 and idx > 0:
                print(f"[Visualization!] Iteration : {idx} || Generator loss: {g_loss.item()} || Discriminator loss: {d_loss.item()}")

                imsave(real_x, f"./sample/real_{epoch}_epoch_{idx}_iter.png")
                imsave(fake_x, f"./sample/fake_{epoch}_epoch_{idx}_iter.png")
                
        ''' Save model '''
        print(f"Epoch : {epoch} || time : {time.time() - start_time} || Generator loss : {g_loss.item()} || Discriminator loss: {d_loss.item()}")
        if args.use_vae:
            torch.save(
                {
                    'netG': netG.state_dict(),
                    'netD': netD.state_dict(),
                    'netE': netE.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_D': optimizer_D.state_dict(),
                    'g_loss': g_loss,
                    'd_loss': d_loss,
                    'args': args,
                },
                f"./checkpoint/checkpoint_useVAE_{epoch}epoch_{i}iter.pt"
            )
        else:
            torch.save(
                {
                    'netG': netG.state_dict(),
                    'netD': netD.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_D': optimizer_D.state_dict(),
                    'g_loss': g_loss,
                    'd_loss': d_loss,
                    'args': args,
                },
                f"./checkpoint/checkpoint_{epoch}epoch_{i}iter.pt"
            )

        # update learning rate
        if epoch > args.niter:
                lrd = args.lr / args.niter_decay
                new_lr = old_lr - lrd
        else:
            new_lr = old_lr

        if new_lr != old_lr:
            if args.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            
            print('update learning rate: %f -> %f' % (old_lr, new_lr))
            old_lr = new_lr
