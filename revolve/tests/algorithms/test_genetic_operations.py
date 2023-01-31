import pytest
from typing import List, Tuple
from random import seed
from revolve.architectures import MLPChromosome, Conv2DChromosome
from revolve.algorithms.evolutionary_orperations import mutation, tournament_selection, uniform_crossover


def test_tournament_selection(generation_data):
    parent1, parent2 = tournament_selection(generation_data, 3)
    assert parent1 in [i[0] for i in generation_data]
    assert parent2 in [i[0] for i in generation_data]


def test_uniform_crossover_mlp(mlp_models, mlp_strategy):
    parent1, parent2 = mlp_models[:2]
    offspring = uniform_crossover(parent1, parent2, 0.9, mlp_strategy)
    assert isinstance(offspring, MLPChromosome)
    for i in offspring.genes:
        assert i in parent1.genes or i in parent2.genes


def test_uniform_crossover_conv2d(conv2d_models, conv2d_strategy):
    parent1, parent2 = conv2d_models[:2]
    offspring = uniform_crossover(parent1, parent2, 0.9, conv2d_strategy)
    assert isinstance(offspring, Conv2DChromosome)
    for i in offspring.genes:
        assert i in parent1.genes or i in parent2.genes


# def test_mutation_mlp(mlp_models):
#     """
#     IMPLEMENT THIS METHOD
#     """
#     offspring = mlp_models[0]
#     learnable_parameters = [1, 2, 3]
#     mutation(offspring, 0.5, learnable_parameters)
#     for i in offspring.genes:
#         assert i in learnable_parameters

