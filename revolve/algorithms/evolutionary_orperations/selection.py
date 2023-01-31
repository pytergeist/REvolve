import random
from typing import Tuple


def tournament_selection(generation_data, size: int) -> Tuple:
    """
    Selects two parents for breeding using tournament selection.

    Parameters:
    generation_data (list): The data for the current generation.

    Returns:
    Tuple: A tuple of the two selected parents.
    """

    competitors = [random.choice(generation_data) for _ in range(size)]
    parent1 = min(competitors, key=lambda x: x[1])
    competitors.remove(parent1)
    parent2 = min(competitors, key=lambda x: x[1])
    return parent1[0], parent2[0]


def roulette_wheel_selection(generation_data) -> Tuple:
    """
    Selects two parents for breeding using roulette wheel selection.

    Parameters:
    generation_data (list): The data for the current generation.

    Returns:
    Tuple: A tuple of the two selected parents.
    """

    total_fitness = sum(data[1] for data in generation_data)
    wheel = []
    current = 0

    for data in generation_data:
        current += data[1]
        wheel.append((data[0], current / total_fitness))

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


def stochastic_universal_selection(generation_data) -> Tuple:
    """
    Selects two parents for breeding using Stochastic Universal Sampling.

    Parameters:
    generation_data (list): The data for the current generation.

    Returns:
    Tuple: A tuple of the two selected parents.
    """

    total_fitness = sum(data[1] for data in generation_data)
    step = total_fitness / len(generation_data)
    start = random.uniform(0, step)
    pointers = [start + i * step for i in range(len(generation_data))]
    wheel = [(data[0], data[1]) for data in generation_data]
    wheel.sort(key=lambda x: x[1])

    selected = []
    i = 0
    for pointer in pointers:
        while wheel[i][1] < pointer:
            pointer -= wheel[i][1]
            i += 1
        selected.append(wheel[i][0])

    parent1, parent2 = random.sample(selected, 2)
    return parent1, parent2

