import random
from typing import List, Union
from revolve.architectures.conv2d_chromosome import Conv2DChromosome
from revolve.architectures import MLPChromosome


def mutation(
    offspring: Union[MLPChromosome, Conv2DChromosome],
    probability: float,
    parameters: dict,
):
    """
    Performs mutation on the given offspring.

    Parameters:
    offspring (list): The offspring to be mutated.
    """

    for idx in range(len(offspring.genes)):
        if random.uniform(0, 1) <= probability:
            offspring.genes[idx].mutate(parameters)

    return offspring
