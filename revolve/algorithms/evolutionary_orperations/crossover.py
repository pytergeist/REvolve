import random
from typing import Union
from revolve.architectures.chromosomes import MLPChromosome, Conv2DChromosome
from revolve.architectures.strategies import MLPStrategy, Conv2DStrategy


def uniform_crossover(
        parent1: Union[MLPChromosome, Conv2DChromosome],
        parent2: Union[MLPChromosome, Conv2DChromosome],
        probability: float, strategy: Union[MLPStrategy, Conv2DStrategy]
        ) -> object:
    """
    Performs uniform crossover on the given parents to produce an offspring.

    Parameters:
    parent1 (list): The first parent.
    parent2 (list): The second parent.

    Returns:
    list: The offspring produced by the crossover.
    """

    offspring = []
    for i in range(len(parent1.genes)):
        if random.random() < probability:
            offspring.append(parent1.genes[i])
        else:
            offspring.append(parent2.genes[i])

    return strategy.create_new_chromosome(genes=offspring)
