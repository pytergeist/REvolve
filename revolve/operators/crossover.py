"""
File containing functions for crossover evolutionary operations
"""

import random
from typing import Tuple, Union
from copy import deepcopy
from revolve.architectures.chromosomes import MLPChromosome, Conv2DChromosome

chromosome_type = Union[MLPChromosome, Conv2DChromosome]


def uniform_crossover(
    parents: Tuple[chromosome_type, chromosome_type],
    probability: float = 0.5,
) -> Union[MLPChromosome, Conv2DChromosome]:
    """
    Performs uniform crossover on the given parents to produce an offspring.

    Parameters:
    parent1 (BaseChromosome): The first parent.
    parent2 (BaseChromosome): The second parent.
    probability (float):

    Returns:
    list: The offspring produced by the crossover.
    """

    parent1, parent2 = parents

    offspring = deepcopy(parent1)

    offspring.genes = [
        parent1.genes[i] if random.random() < probability else parent2.genes[i]
        for i in range(len(parent1.genes))
    ]

    return offspring
