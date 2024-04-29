import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

from models import *
import torch
from torchvision import transforms, datasets
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from PIL import ImageFile
from sklearn.manifold import TSNE
from torchvision.datasets import CIFAR10, CelebA

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='./configs/vae_vis.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

params = config['exp_params']

model = vae_models[config['model_params']['name']](**config['model_params'])

model_dict = torch.load('checkpoints/VanillaVAE (CelebA, LD=40, LR=0.001).ckpt',
                        map_location=torch.device('cpu'))

newdict = dict()
for k, v in model_dict['state_dict'].items():
    knew = k.split('.', 1)[1:][0]
    newdict[knew] = v

model.load_state_dict(newdict)
model.eval()


def data_transforms():
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    SetScale = transforms.Lambda(lambda X: X / X.sum(0).expand_as(X))

    if params['dataset'] == 'cifar10':
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(148),
                                        transforms.Resize(params['img_size']),
                                        transforms.ToTensor(),
                                        SetRange])
    elif params['dataset'] == 'celeba':
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(148),
                                        transforms.Resize(params['img_size']),
                                        transforms.ToTensor(),
                                        SetRange])
    elif params['dataset'] == 'wikiart':
        transform = transforms.Compose([transforms.Resize(64),
                                        transforms.RandomCrop(64),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    else:
        raise ValueError('Undefined dataset type')
    return transform


def get_dataloader():
    transform = data_transforms()

    if params['dataset'] == 'cifar10':
        sample_dataloader = DataLoader(CIFAR10(root=params['data_path'],
                                               train=False,
                                               transform=transform,
                                               download=False),
                                       batch_size=144,
                                       num_workers=12,
                                       shuffle=True,
                                       drop_last=True)
        num_val_imgs = len(sample_dataloader)
    elif params['dataset'] == 'celeba':
        sample_dataloader = DataLoader(CelebA(root=params['data_path'],
                                              split="test",
                                              transform=transform,
                                              download=False),
                                       num_workers=12,
                                       batch_size=144,
                                       shuffle=True,
                                       drop_last=True)
        num_val_imgs = len(sample_dataloader)
    elif params['dataset'] == 'wikiart':
        main_dataset = datasets.ImageFolder(root=params['data_path'] + 'wikiart',
                                            transform=transform)
        train_size = int(0.8 * len(main_dataset))
        test_size = len(main_dataset) - train_size
        _, test_dataset = torch.utils.data.random_split(main_dataset, [train_size, test_size],
                                                        generator=torch.Generator().manual_seed(42))
        sample_dataloader = DataLoader(test_dataset,
                                       num_workers=12,
                                       batch_size=144,
                                       shuffle=True,
                                       drop_last=True)
        num_val_imgs = len(sample_dataloader)
    else:
        raise ValueError('Undefined dataset type')

    return sample_dataloader


try:
    embed = np.load('vis_embed_arr.npy')
    label_emb = np.load('vis_label_embed_arr.npy')
except:
    dataloader = get_dataloader()
    embed = []
    label_emb = []
    for inputs, labels in dataloader:
        with torch.no_grad():
            if config['exp_params']['dataset'] == 'wikiart':
                num_classes = 27
            else:
                num_classes = -1
            # for the colors
            label_emb.append(labels.numpy())
            # for the embedding
            labels = torch.nn.functional.one_hot(labels, num_classes)
            embed.append(model.embed(inputs, labels=labels).numpy())

    embed = np.array(embed).reshape(-1, model.latent_dim)
    label_emb = np.array(label_emb)
    label_emb = label_emb.reshape(-1, label_emb.shape[-1])

    np.save('vis_embed_arr', embed)
    np.save('vis_label_embed_arr', label_emb)


black_hair_idx = np.argwhere(label_emb[:, 8] == 1)
black_hair_x = embed[black_hair_idx].squeeze()
black_hair_y = np.ones(black_hair_x.shape[0]) * 0

blonde_hair_idx = np.argwhere(label_emb[:, 9] == 1)
blonde_hair_x = embed[blonde_hair_idx].squeeze()
blonde_hair_y = np.ones(blonde_hair_x.shape[0]) * 1

brown_hair_idx = np.argwhere(label_emb[:, 11] == 1)
brown_hair_x = embed[brown_hair_idx].squeeze()
brown_hair_y = np.ones(brown_hair_x.shape[0]) * 2

gray_hair_idx = np.argwhere(label_emb[:, 17] == 1)
gray_hair_x = embed[gray_hair_idx].squeeze()
gray_hair_y = np.ones(gray_hair_x.shape[0]) * 3

hair_x = np.vstack([black_hair_x, blonde_hair_x, brown_hair_x, gray_hair_x])
hair_y = np.hstack([black_hair_y, blonde_hair_y, brown_hair_y, gray_hair_y]).astype(int)

class_labels = hair_y
class_names = ['black', 'blonde', 'brown', 'grey']
colors = np.array(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'])
class_labels = colors[class_labels]

if model.latent_dim == 2:
    scatter = plt.scatter(embed[:, 0], embed[:, 1], c=label_emb)
    plt.savefig('embed' + config['model_params']['name'] + str(config['model_params']['latent_dim']) + 'LD' + '.png')
elif model.latent_dim == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(hair_x[:, 0], hair_x[:, 1], hair_x[:, 2], c=class_labels, s=0.1)
    plt.title('3D plot of InfoVAE latent space (Hair Color)')
    plt.savefig('embed' + config['model_params']['name'] + str(config['model_params']['latent_dim']) + 'LD' + '.png')
else:
    try:
        embed_tsne = np.load('VAE_TSNE_40_HAIR.npy')
    except:
        embed_tsne = TSNE(n_components=2, random_state=2).fit_transform(hair_x)
        np.save('VAE_TSNE_40_HAIR', embed_tsne)

    scatter = plt.scatter(embed_tsne[:, 0], embed_tsne[:, 1], c=class_labels, s=1)
    handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in colors]
    plt.title('tSNE for 40-dimensional InfoVAE latent embeddings (Hair Color)')
    plt.legend(handlelist, class_names, loc='lower left')
    plt.savefig('embed' + config['model_params']['name'] + str(config['model_params']['latent_dim']) + 'LD' + '_HAIR.png')
