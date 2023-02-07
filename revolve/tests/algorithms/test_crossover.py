import pytest
from revolve.algorithms.evolutionary_orperations import uniform_crossover
from revolve.architectures import MLPChromosome, Conv2DChromosome


class Parent:
    def __init__(self, genes):
        self.genes = genes


@pytest.mark.parametrize(
    "strategy, population, expected_chromosome",
    [
        ("mlp_strategy", "mlp_population", MLPChromosome),
        ("conv2d_strategy", "conv_population", Conv2DChromosome),
    ],
)
def test_uniform_crossover(strategy, population, expected_chromosome, request):
    strategy = request.getfixturevalue(strategy)
    population = request.getfixturevalue(population)

    # test crossover returns expected chromosome with mock population
    crossover_offspring = uniform_crossover(population[:2], 0.5)

    assert isinstance(crossover_offspring, expected_chromosome)

    # test crossover returns expected chromosome with mock population
    crossover_offspring = uniform_crossover(population[:2], 0.5)
    assert isinstance(crossover_offspring, expected_chromosome)

    # test crossover returns expected gene sequence
    parent1 = Parent([1, 1, 1, 1])
    parent2 = Parent([0, 0, 0, 0])

    mock_crossover_offspring = uniform_crossover((parent1, parent2), 1)
    assert mock_crossover_offspring.genes != parent2.genes
    mock_crossover_offspring = uniform_crossover((parent1, parent2), 0)
    assert mock_crossover_offspring.genes != parent1.genes
