import random
from revolve.architectures.base import BaseChromosome


def mutation(
    offspring: BaseChromosome,
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
