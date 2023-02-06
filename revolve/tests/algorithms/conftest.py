import pytest
from revolve.architectures.base import Chromosome
from unittest import mock


@pytest.fixture
def mlp_population(mlp_chromosome):
    population = [mlp_chromosome for _ in range(10)]
    for idx in range(len(population)):
        population[idx].loss = float(idx)
        population[idx].metric = float(idx)
    return population


@pytest.fixture
def conv_population(conv2d_chromosome):
    population = [conv2d_chromosome for _ in range(10)]
    for idx, chromosome in enumerate(population):
        chromosome.loss = idx
    return population


@pytest.fixture(autouse=True)
def operations():
    operations = mock.MagicMock()
    operations.selection = mock.MagicMock(return_value=(Chromosome, Chromosome))
    operations.crossover = mock.MagicMock(return_value=Chromosome)
    operations.mutation = mock.MagicMock(return_value=Chromosome)
    return operations
