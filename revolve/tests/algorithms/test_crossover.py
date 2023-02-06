import pytest
from revolve.algorithms.evolutionary_orperations import uniform_crossover
from revolve.architectures import MLPChromosome, Conv2DChromosome


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

    crossover_offspring = uniform_crossover(population[:2], 0.5, strategy)

    assert isinstance(crossover_offspring, expected_chromosome)
