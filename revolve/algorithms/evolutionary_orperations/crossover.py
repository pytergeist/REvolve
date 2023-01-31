import random


def uniform_crossover(parent1: object, parent2: object, probability: float, strategy: object) -> object:
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
