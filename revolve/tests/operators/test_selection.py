import pytest
from revolve.operators import (
    tournament_selection,
    roulette_wheel_selection,
)
from revolve.architectures.chromosomes import MLPChromosome, Conv2DChromosome


@pytest.mark.parametrize(
    "population, expected_chromosome",
    [("mlp_population", MLPChromosome), ("conv_population", Conv2DChromosome)],
)
def test_tournament_selection(population, expected_chromosome, request):
    population = request.getfixturevalue(population)
    parent1, parent2 = tournament_selection(population, 3)
    assert isinstance(parent1, expected_chromosome)
    assert isinstance(parent2, expected_chromosome)
    assert parent1 in population
    assert parent2 in population


@pytest.mark.parametrize(
    "population, expected_chromosome",
    [("mlp_population", MLPChromosome), ("conv_population", Conv2DChromosome)],
)
def test_roulette_wheel_selection(population, expected_chromosome, request):
    # Test binary search inner function
    population = request.getfixturevalue(population)
    parent1, parent2 = roulette_wheel_selection(population)
    assert isinstance(parent1, expected_chromosome)
    assert isinstance(parent2, expected_chromosome)
    assert parent1 in population
    assert parent2 in population
