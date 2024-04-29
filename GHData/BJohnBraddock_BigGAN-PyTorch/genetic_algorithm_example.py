from random import choices, random, randrange, randint
from typing import Callable, List, Tuple
import numpy as np
from scipy.stats import truncnorm
import torch
import torch.nn as nn
import torch.nn.functional as F

Population = List[torch.Tensor]

def truncated_noise_sample(batch_size: int=1, dim_z: int=128, truncation: float =1., seed=None):
    """ Create a truncated noise vector.
        Params:
            batch_size: batch size.
            dim_z: dimension of z
            truncation: truncation value to use
            seed: seed for the random generator
        Output:
            array of shape (batch_size, dim_z)
    """
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values

def generate_class_vector(num_classes: int) -> torch.Tensor:
    return torch.from_numpy(truncnorm.rvs(0, 1, size=num_classes))


def generate_population(size: int, num_classes: int) -> List[torch.Tensor]:
    return [generate_class_vector(num_classes) for _ in range(size)]

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
    return torch.cat((a[0:p], b[p:]), dim=0), torch.cat((b[0:p], a[p:]), dim=0)


def VCA_fitness_function(class_vector: torch.Tensor, latent_vector: torch.Tensor, generator: nn.Module, VCA: nn.Module) -> float:
    
    def fitness(class_vector: torch.Tensor) -> float:
        generator.eval()
        VCA.eval()

        G_z  = generator(class_vector, latent_vector)
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





device = 'gpu' if torch.cuda.is_available() else 'cpu'

num_classes = 100
batch_size = 10

generation_limit = 1000
fitness_limit = 90.0

population = generate_population(batch_size, num_classes)

for i in range(generation_limit):
    population = sort_population(population, dummy_fitness)


    if dummy_fitness(population[0]) >= fitness_limit:
        break

    next_generation = population[0:2]

    for j in range(int(len(population)/2)-1):
        
        parents = select_parents(population, dummy_fitness)

        offspring_a, offspring_b = single_point_crossover(parents[0], parents[1])

        offspring_a = mutate_class_vector(offspring_a)
        offspring_b = mutate_class_vector(offspring_b)

        next_generation += [offspring_a, offspring_b]
    
    population = next_generation


print(f"Final Population ({i} generations")
print(population)

print(f"Best ({dummy_fitness(population[0])})")
print(population[0])

