import argparse
import cv2
import numpy as np
import torch
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm

from gan.utils import d_loop, g_loop, g_sample
from dataset.image_dataset import AddGaussianNoise, ImageDataset, make_dataloader
from dataset.utils import decode_img, array_yxc2cyx
from gan.generator import ImageGenerator
from gan.discriminator import ImageDiscriminator
from utils import get_readable_timestamp, GrayToRgb


def train(generator, discriminator, train_loader, data_is_labeled, optimizer_gen, optimizer_discr, epoch_id=0,
          g_steps=1, d_steps=1, device="cpu", autosave_period=None, valid_period=None,
          tb_writer=None, transform=None):

    with tqdm(total=len(train_loader) * train_loader.batch_size,
              desc=f'Epoch {epoch_id + 1}',
              unit='image') as pbar:
        for batch_idx, batch_data in enumerate(train_loader):
            real_imgs = batch_data[0] if data_is_labeled else batch_data

            d_infos = []
            for d_index in range(d_steps):
                d_info = d_loop(generator=generator, discriminator=discriminator, d_optimizer=optimizer_discr,
                                real_batch=real_imgs, cuda=device == "cuda")
                d_infos.append(d_info)
            d_infos = np.mean(d_infos, 0)
            d_real_loss, d_fake_loss, d_real_accuracy, d_fake_accuracy = d_infos

            g_infos = []
            for g_index in range(g_steps):
                g_info = g_loop(generator=generator, discriminator=discriminator, g_optimizer=optimizer_gen,
                                d_optimizer=optimizer_discr, real_batch=real_imgs, cuda=device == "cuda")
                g_infos.append(g_info)
            g_infos = np.mean(g_infos, 0)
            g_loss, g_accuracy = g_infos

            global_step = epoch_id * len(train_loader) + batch_idx
            if tb_writer:
                tb_writer.add_scalar("Loss/Discriminator", d_real_loss + d_fake_loss, global_step)
                tb_writer.add_scalar("Loss/DiscriminatorFake", d_fake_loss, global_step)
                tb_writer.add_scalar("Loss/DiscriminatorReal", d_real_loss, global_step)
                tb_writer.add_scalar("Loss/Generator", g_loss, global_step)
                tb_writer.add_scalar("Loss/Total", g_loss + d_fake_loss + d_real_loss, global_step)
                tb_writer.add_scalar("Accuracy/DiscrReal", d_real_accuracy, global_step)
                tb_writer.add_scalar("Accuracy/DiscrFake", d_fake_accuracy, global_step)
                tb_writer.add_scalar("Accuracy/GenFake", g_accuracy, global_step)

            if valid_period:
                if (batch_idx + 1) % valid_period == 0:
                    with torch.no_grad():
                        fake_imgs = g_sample(generator=generator, batch_size=len(real_imgs), cuda=device == "cuda")
                        gen_imgs = []
                        for i, img in enumerate(fake_imgs):
                            img = decode_img(img)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = array_yxc2cyx(img)
                            gen_imgs.append(torch.from_numpy(img))

                        target_imgs = []
                        for i, img in enumerate(real_imgs[:min(len(real_imgs), 8)]):
                            img = decode_img(img.numpy())
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = array_yxc2cyx(img)
                            target_imgs.append(torch.from_numpy(img))

                        if tb_writer:
                            img_grid_gen = torchvision.utils.make_grid(gen_imgs)
                            tb_writer.add_image('Valid/Generated', img_tensor=img_grid_gen,
                                                global_step=global_step, dataformats='CHW')
                            img_grid_tgt = torchvision.utils.make_grid(target_imgs)
                            tb_writer.add_image('Valid/Real', img_tensor=img_grid_tgt,
                                                global_step=global_step, dataformats='CHW')

            if autosave_period is not None:
                if (batch_idx + 1) % autosave_period == 0:
                    model_name = str(generator) + "_" + get_readable_timestamp() + "_epoch_" + \
                               str(epoch_id) + "_batch_" + str(batch_idx) + ".pt"
                    torch.save(generator.state_dict(), model_name)
                    print(model_name, " has been saved")
                    model_name = str(discriminator) + "_" + get_readable_timestamp() + "_epoch_" + \
                               str(epoch_id) + "_batch_" + str(batch_idx) + ".pt"
                    torch.save(discriminator.state_dict(), model_name)
                    print(model_name, " has been saved")

            pbar.update(train_loader.batch_size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs")
    parser.add_argument("--batch-train", type=int, default=32,
                        help="Size of batch for training")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Target device: cpu/cuda")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="Type of optmizer")
    parser.add_argument("--autosave-period", type=int, default=0,
                        help="Period of model autosave")
    parser.add_argument("--autosave-period-unit", type=str, default="e",
                        help="Units for autosave (e/b)")
    parser.add_argument("--valid-period", type=int, default=10,
                        help="Period of validation")
    parser.add_argument("--valid-period-unit", type=str, default="e",
                        help="Units for validation (e/b)")
    parser.add_argument("--pretrained_gen", type=str,
                        help="Abs path to pretrained generator")
    parser.add_argument("--pretrained_disc", type=str,
                        help="Abs path to pretrained discriminator")
    parser.add_argument("--scheduler", type=int, default=0,
                        help="Use lr scheduler or not")
    parser.add_argument("--l2", type=float, default=0,
                        help="L2 reularization coefficient")
    parser.add_argument("--noise-mean", type=float, default=None,
                        help="Gaussian noise mean")
    parser.add_argument("--noise-std", type=float, default=None,
                        help="Gaussian noise std")
    parser.add_argument("--dataset", type=str,
                        default="mnist",
                        help="Abs path to dataset")
    return parser.parse_args()


def get_optimizer(optim, model, lr, l2):
    if optim == "adam":
        return Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=l2)
    elif optim == "sgd":
        return SGD(model.parameters(), lr=lr, weight_decay=l2)
    else:
        return None


if __name__ == "__main__":
    args = parse_args()

    gen_model = ImageGenerator()
    gen_model = gen_model.to(args.device)
    if args.pretrained_gen:
        gen_model.load_state_dict(torch.load(args.pretrained_gen))

    discr_model = ImageDiscriminator()
    discr_model = discr_model.to(args.device)
    if args.pretrained_disc:
        discr_model.load_state_dict(torch.load(args.pretrained_disc))

    train_dloader = None
    transform = None
    dset_is_labeled = False
    if (args.noise_mean is not None) and (args.noise_std is not None):
        transform = AddGaussianNoise(mean=args.noise_mean, std=args.noise_std)
    if args.dataset == "mnist":
        train_dloader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('/home/igor/datasets/mnist/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Resize(size=(64, 64)),
                                           torchvision.transforms.Normalize(
                                               mean=(0.5,), std=(0.5,)),
                                           GrayToRgb(),
                                       ])),
            batch_size=args.batch_train, shuffle=True)
        dset_is_labeled = True
    else:
        dataset = ImageDataset(args.dataset, target_size=(64, 64), transform=transform)
        train_dloader = make_dataloader(dataset, batch_size=args.batch_train, shuffle_dataset=True)

    optimizer_g = get_optimizer(args.optimizer, gen_model, lr=args.learning_rate, l2=args.l2)
    optimizer_d = get_optimizer(args.optimizer, discr_model, lr=args.learning_rate, l2=args.l2)
    tboard_writer = SummaryWriter()

    try:
        for e in range(args.epochs):
            train(gen_model, discr_model, train_dloader, dset_is_labeled, optimizer_g, optimizer_d, e,
                  device=args.device, autosave_period=None, valid_period=args.valid_period, tb_writer=tboard_writer,
                  transform=transform)
        model_name = "pretrained_models/" + str(gen_model) + "_completed_" + get_readable_timestamp() + ".pt"
        torch.save(gen_model.state_dict(), model_name)
        print("Training completed. Final model " + model_name + " has been saved")
        model_name = "pretrained_models/" + str(discr_model) + "_completed_" + get_readable_timestamp() + ".pt"
        torch.save(discr_model.state_dict(), model_name)
        print("Training completed. Final model " + model_name + " has been saved")
    except KeyboardInterrupt:
        model_name = "pretrained_models/" + str(gen_model) + "_completed_" + get_readable_timestamp() + ".pt"
        torch.save(gen_model.state_dict(), model_name)
        print("Training completed. Final model " + model_name + " has been saved")
        model_name = "pretrained_models/" + str(discr_model) + "_completed_" + get_readable_timestamp() + ".pt"
        torch.save(discr_model.state_dict(), model_name)
        print("Training completed. Final model " + model_name + " has been saved")
