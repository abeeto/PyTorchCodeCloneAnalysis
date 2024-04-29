import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import random as pyrandom
from scipy.stats import truncnorm
from argparse import ArgumentParser

from pytorch_pretrained_biggan import BigGAN
from genetic_algorithm_example import truncated_noise_sample

from models.Amy_IntermediateRoad import Amy_IntermediateRoad
from vca_utils import load_checkpoint
from genetic_fns import generate_class_vector, generate_population, mutate_class_vector, single_point_crossover, VCA_fitness_function, sort_population, select_parents

from typing import List

import neptune.new as neptune
from neptune.new.types import File
import Constants


def run(config):

    neptune_run = neptune.init(project='bjohnbraddock/BigGAN-VCA-Evolution', api_token=Constants.NEPTUNE_API_KEY, source_files=['genetic_algorithm_GAN.py', 'genetic_fns.py'])

    neptune_run['config/model'] = 'BigGAN'
    neptune_run['config/criterion'] = 'VCA'
    neptune_run['config/params'] = config

    assert torch.cuda.is_available(), 'Torch could not find CUDA enabled GPU'
    device = 'cuda'

    # Seed RNG
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    pyrandom.seed(seed)

    torch.backends.cudnn.benchmark = True

    print(f"Loading BigGAN Generator...")
    G = BigGAN.from_pretrained('biggan-deep-256', cache_dir=config['biggan_cache_dir']).to(device)
    
    print(f"Loading VCA from {config['vca_filepath']}")
    VCA = Amy_IntermediateRoad( lowfea_VGGlayer=10, highfea_VGGlayer=36, is_highroad_only=False, is_gist=False)
    VCA = load_checkpoint(VCA, config['vca_filepath'])
    VCA = VCA.to(device)


    latent_vector = config['truncation'] * truncnorm.rvs(-2, 2, size=(config['dim_z'])).astype(np.float64)
    latent_vector = torch.from_numpy(latent_vector).to(device, dtype=torch.float32).unsqueeze(0)

    def log_images(population: List[torch.Tensor]):
        cut_pop = population[: min(config['population_size'], config['image_log_size'])]
        G_z = G(latent_vector.repeat((len(cut_pop),1)), torch.stack(cut_pop).squeeze(), truncation=config['truncation'])

        G_z = F.interpolate(G_z, size=224)
        VCA_G_z = VCA(G_z)

        Gz_grid = torchvision.utils.make_grid(G_z, normalize=True)
        Gz_grid = torch.permute(Gz_grid, (1,2,0))

        neptune_run['train/images'].log(File.as_image(Gz_grid.cpu()))
        neptune_run['train/vca_scores'].log(VCA_G_z)

    # Evolution
    fitness_func = VCA_fitness_function(latent_vector, G, VCA, truncation=config['truncation'])
    population = generate_population(config['population_size'], config['num_classes'])
    population = [vec.to(device, dtype=torch.float32) for vec in population]

    for generation in range(config['generation_limit']):
        G.eval()
        VCA.eval()
        VCA_population = []
        for vec in population:
            G_z  = G(latent_vector,vec, truncation=config['truncation'])
            G_z = F.interpolate(G_z, size=224)
            VCA_population.append(VCA(G_z).item())

        sort_idx = np.argsort(VCA_population)
        best_idx = sort_idx[0]

        # population = sort_population(population, fitness_func)

        if VCA_population[best_idx] >= config['fitness_limit']:
            break

        
        neptune_run['training/vca_mean'].log(np.mean(VCA_population))
        neptune_run['training/vca_best'].log(VCA_population[best_idx])
        if not(generation % config['log_every']):
            print('Generation: {}    Best VCA_G_z {}'.format(generation, VCA_population[best_idx]))
            log_images(population)

        
        next_generation = [population[sort_idx[0]], population[sort_idx[1]]]

        for j in range(int(len(population)/2) -1):
            parents = pyrandom.choices(population, VCA_population, k=2)

            offspring_a, offspring_b = parents[0], parents[1]

            offspring_a = mutate_class_vector(offspring_a, num=4, probability=config['mutate_probability'])
            offspring_b = mutate_class_vector(offspring_b, num=4, probability=config['mutate_probability'])

            next_generation += [offspring_a, offspring_b]
        
        population = next_generation
    
    log_images(population)
    


def main():

    parser = ArgumentParser(description='Parser for simplified refactor')
    parser.add_argument(
    '--seed', type=int, default=0,
    help='Random seed to use; affects both initialization and '
        ' dataloading. (default: %(default)s)')
    parser.add_argument(
    '--vca_filepath', type=str, default='',
    help='Relative filepath to trained VCA model .pth (default: %(default)s)'    )
    parser.add_argument(
    '--population_size', type=int, default=64,
    help='Number of vectors in each generation (default: %(default)s)')
    parser.add_argument(
    '--dim_z', type=int, default=128,
    help='Noise dimensionality: %(default)s)')
    parser.add_argument(
    '--num_classes', type=int, default=1000,
    help='Number of classes: %(default)s')
    parser.add_argument(
    '--truncation', type=float, default=0.4,
    help='BigGAN truncation parameter for sampling normal distr. (default: %(default)s)')
    parser.add_argument(
    '--generation_limit', type=int, default=200,
    help='Max generations to run if fitness limit is not met (default: %(default)s)')
    parser.add_argument(
    '--fitness_limit', type=float, default=1.0,
    help='Fitness (VCA score) at which to stop training (default: %(default)s)')
    parser.add_argument(
    '--mutate_probability', type=float, default=0.1,
    help='Probability of a mutation occuring when offspring is created from crossover (default: %(default)s)')
    parser.add_argument(
    '--log_every', type=int, default=25,
    help='Log images and metrics every X iterations (default: %(default)s')
    parser.add_argument(
    '--image_log_size', type=int, default=16,
    help='Number of images to save and log in a grid (default: %(default)s')
    parser.add_argument(
    '--biggan_cache_dir', type=str, default='/blue/ruogu.fang/bjohn.braddock/BigGAN/pretrained',
    help='Where to cache BigGAN from TFHUB (default: %(default)s)'    )
    # parser.add_argument(
    # '--train_unpleasant', action='store_true', default=False,
    # help='Set to optimize VCA response of 0 (unpleasant) (default: %(default)s)')

    
    config = vars(parser.parse_args())

    print(config)
    run(config)

if __name__ == '__main__':
    main()