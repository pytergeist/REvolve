"""
File containing functions for mutation evolutionary operations
"""

import random
from typing import Union
from revolve.architectures.chromosomes import MLPChromosome, Conv2DChromosome


def random_mutation(
    offspring: Union[MLPChromosome, Conv2DChromosome],
    probability: float,
    parameters: dict,
):
    """
    Performs mutation on the given offspring.

    Parameters:
    offspring (list): The offspring to be mutated.
    """

    for idx, _ in enumerate(offspring.genes):
        if random.uniform(0, 1) <= probability:
            offspring.genes[idx].mutate(idx, parameters)

    return offspring
