import random
from typing import Tuple


def tournament_selection(population, size: int) -> Tuple:
    """
    Selects two parents for breeding using tournament selection.

    Parameters:
    generation_data (list): The data for the current generation.

    Returns:
    Tuple: A tuple of the two selected parents.
    """
    fitness = [(chromosome, chromosome.loss) for chromosome in population]
    competitors = [random.choice(fitness) for _ in range(size)]
    parent1 = min(competitors, key=lambda x: x[1])
    competitors.remove(parent1)
    parent2 = min(competitors, key=lambda x: x[1])
    return parent1[0], parent2[0]


def roulette_wheel_selection(population) -> Tuple:
    """
    Selects two parents for breeding using roulette wheel selection.

    Parameters:
    generation_data (list): The data for the current generation.

    Returns:
    Tuple: A tuple of the two selected parents.
    """

    total_fitness = sum(chromosome.loss for chromosome in population)
    wheel = []
    current = 0

    for chromosome in population:
        current += chromosome.loss
        wheel.append((chromosome, current / total_fitness))

    def binary_search(value):
        low, high = 0, len(wheel) - 1
        while low <= high:
            mid = (low + high) // 2
            if wheel[mid][1] >= value:
                high = mid - 1
            else:
                low = mid + 1
        return wheel[low][0]

    parent1 = binary_search(random.uniform(0, 1))
    parent2 = binary_search(random.uniform(0, 1))

    return parent1, parent2
