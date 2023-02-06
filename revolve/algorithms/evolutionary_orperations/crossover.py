import random
from typing import Tuple
from revolve.architectures.base import Chromosome, Strategy


def uniform_crossover(
    parents: Tuple[Chromosome],
    probability: float,
    strategy: Strategy,
) -> object:
    """
    Performs uniform crossover on the given parents to produce an offspring.

    Parameters:
    parent1 (list): The first parent.
    parent2 (list): The second parent.

    Returns:
    list: The offspring produced by the crossover.
    """

    parent1, parent2 = parents

    offspring = []
    for i in range(len(parent1.genes)):
        if random.random() < probability:
            offspring.append(parent1.genes[i])
        else:
            offspring.append(parent2.genes[i])

    return strategy.create_new_chromosome(genes=offspring)
