import argparse
from collections import OrderedDict
import numpy as np
import os
import pickle as pkl
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import yaml


from models import *
from vae_experiment import VariationalAutoencoderExperiment


class Embedder:

    def __init__(self, vae_model: BaseVAE, params: dict, checkpoint_path):
        self.vae_model = vae_model
        self.params = params
        self.val_dataloader = self._val_dataloader()

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            old_state_dict = checkpoint['state_dict']
        else:
            raise FileNotFoundError("Invalid checkpoint path!")

        gpu_name = 'cuda:{}'.format(self.params['gpu_idx'])
        self.device = torch.device(gpu_name if torch.cuda.is_available() else 'cpu')

        self._prepare_model(old_state_dict=old_state_dict)

    def _prepare_model(self, old_state_dict: OrderedDict):
        new_state_dict = OrderedDict()
        for old_key in old_state_dict:
            new_key = old_key[6:]
            new_state_dict[new_key] = old_state_dict[old_key]
            print('Loading:', new_key)
        assert len(self.vae_model.state_dict()) == len(new_state_dict)
        self.vae_model.load_state_dict(new_state_dict)
        print('Loading successful!')
        self.vae_model.eval()
        self.vae_model.to(self.device)

    def save_embeddings(self):
        # TODO: make it possible
        # set up device
        print('Saving Embeddings...')
        means, log_vars = [], []
        with torch.no_grad():
            for (data, _) in self.val_dataloader:
                data = data.to(self.device)
                print(data.shape)
                mu, log_var = self.vae_model.encode(data)
                print(mu.shape)
                print(log_var.shape)
                means.append(mu.cpu().numpy())
                log_vars.append(log_var.cpu().numpy())
        print(len(means))
        print(len(log_vars))
        torch.save(means, 'embeddings_means.pt')
        torch.save(log_vars, 'embeddings_log_vars.pt')
        print('Embeddings Saved...')

    def _val_dataloader(self):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        if self.params['dataset'] == 'cifar10':
            data_path = os.path.join(os.getcwd(), self.params['data_path'])
            dataset = CIFAR10(data_path, train=False, transform=transform, download=False)
            self.val_dataloader = DataLoader(dataset=dataset,
                                             batch_size=self.params['embed_batch_size'],
                                             pin_memory=True,
                                             num_workers=self.params['num_workers'],
                                             drop_last=True)
        else:
            raise ValueError('Undefined dataset type')
        return self.val_dataloader


def main():
    parser = argparse.ArgumentParser('Create embeddings from a trained model')
    parser.add_argument('--config', '-c', dest='filename', metavar='FILE', default='configs/vae.yaml')
    args = parser.parse_args()
    with open(args.filename, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False
    model = vae_models[config['model_params']['name']](**config['model_params'])
    checkpoint_path = 'test.ckpt'

    embedder = Embedder(vae_model=model, params=config['embed_params'],
                        checkpoint_path=checkpoint_path)
    embedder.save_embeddings()


if __name__ == '__main__':
    main()

