from random import choices, random, randrange, randint
from typing import Callable, List, Tuple
import numpy as np
from scipy.stats import truncnorm
import torch
import torch.nn as nn
import torch.nn.functional as F

Population = List[torch.Tensor]

def generate_class_vector(num_classes: int) -> torch.Tensor:
    return F.softmax(torch.from_numpy(truncnorm.rvs(0, 1, size=num_classes)))

def generate_population(size: int, num_classes: int) -> List[torch.Tensor]:
    return [generate_class_vector(num_classes).unsqueeze(0) for _ in range(size)]

def mutate_class_vector(vector: torch.Tensor, num: int = 1, probability: float=0.5) -> torch.Tensor:
    mutated = vector.detach()
    for _ in range(num):
        index = randrange(len(mutated))

        mutated[index] = np.asscalar(truncnorm.rvs(0, 1)) if random() < probability else mutated[index]
    return vector

def single_point_crossover(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if a.size() != b.size():
        raise ValueError("Class Vectors a and b must be of same size")

    length = a.size(0)
    if length < 2:
        return a, b

    p = randint(1, length-1)
    print(f'Crossing over at {p}')
    return torch.cat((a[0:p], b[p:]), dim=0), torch.cat((b[0:p], a[p:]), dim=0)


def VCA_fitness_function(latent_vector: torch.Tensor, generator: nn.Module, VCA: nn.Module, truncation: float) -> float:
    
    def fitness(class_vector: torch.Tensor) -> float:
        generator.eval()
        VCA.eval()

        G_z  = generator(latent_vector, class_vector, truncation=truncation)
        G_z = F.interpolate(G_z, size=224)
        VCA_G_z = VCA(G_z).item()

        return VCA_G_z
    
    return fitness

def dummy_fitness(class_vector: torch.Tensor) -> float:
    return torch.sum(class_vector)

def select_parents(population: Population, fitness_func: Callable[[torch.Tensor], float]):
    return choices(
        population=population,
        weights = [fitness_func(vector) for vector in population],
        k=2
    )

def sort_population(population: Population, fitness_func: Callable[[torch.Tensor], float]) -> Population:
    return sorted(
        population,
        key=fitness_func,
        reverse=True)