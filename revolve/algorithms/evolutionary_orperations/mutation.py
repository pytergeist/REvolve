import random
from typing import List, Union
from revolve.architectures.chromosomes import MLPChromosome, Conv2DChromosome


def mutation(
        offspring: Union[MLPChromosome, Conv2DChromosome], probability: float,
        learnable_parameters: dict,
):
    """
    Performs mutation on the given offspring.

    Parameters:
    offspring (list): The offspring to be mutated.
    """

    for idx in range(len(offspring.genes)):
        if random.uniform(0, 1) <= probability:
            offspring.genes[idx].mutate(learnable_parameters)

    return offspring
