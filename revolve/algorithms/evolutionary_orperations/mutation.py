import random
from typing import List


def mutation(
        offspring: List, probability: float,
        learnable_parameters: List,
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
