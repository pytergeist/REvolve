import pytest
from unittest import mock

from revolve.algorithms.evolutionary_algorithm_elitism import (
    EvolutionaryAlgorithmElitism,
)
from revolve.architectures.chromosomes import MLPChromosome, Conv2DChromosome
from revolve.architectures.base import BaseChromosome


@pytest.mark.parametrize(
    "population, strategy, expected_chromosome",
    [
        ("mlp_population", "mlp_strategy", MLPChromosome),
        ("conv_population", "conv2d_strategy", Conv2DChromosome),
    ],
)
def test_elitism(population, strategy, expected_chromosome, request):
    strategy = request.getfixturevalue(strategy)
    operations = request.getfixturevalue("operations")
    population = request.getfixturevalue(population)
    ea = EvolutionaryAlgorithmElitism(
        strategy=strategy,
        pop_size=10,
        elitism_size=2,
        operations=operations,
    )
    ea.population = population
    assert len(ea.population) == 10
    population, elite_models = ea.elitism(
        population=population,
        elitism_size=2,
        models=[1, 2, 3],
    )
    assert len(population) == 2
    assert len(elite_models) == 2
    assert isinstance(population[0], expected_chromosome)
    assert isinstance(elite_models[0], int)


@pytest.mark.parametrize(
    "population, strategy, expected_chromosome",
    [
        ("mlp_population", "mlp_strategy", MLPChromosome),
        ("conv_population", "conv2d_strategy", Conv2DChromosome),
    ],
)
def test_get_min_fitness(population, strategy, expected_chromosome, request):
    population = request.getfixturevalue(population)
    strategy = request.getfixturevalue(strategy)
    operations = request.getfixturevalue("operations")

    ea = EvolutionaryAlgorithmElitism(
        strategy=strategy,
        pop_size=10,
        elitism_size=2,
        operations=operations,
    )
    ea.population = population
    min_fitness = ea.get_min_fitness(population)
    assert isinstance(min_fitness, expected_chromosome)


@pytest.mark.parametrize(
    "population, strategy, data",
    [
        ("mlp_population", "mlp_strategy", "mock_data"),
        (
            "conv_population",
            "conv2d_strategy",
            "mock_data",
        ),
    ],
)
def test_evolve_population(population, strategy, data, request):
    population = request.getfixturevalue(population)
    strategy = request.getfixturevalue(strategy)
    data = request.getfixturevalue(data)
    operations = request.getfixturevalue("operations")
    model_mock = mock.MagicMock()

    ea = EvolutionaryAlgorithmElitism(
        strategy=strategy,
        pop_size=10,
        elitism_size=2,
        operations=operations,
    )
    ea.get_model_fitness = mock.MagicMock(return_value=model_mock)
    ea.population = population
    best_chromosome = ea.evolve_population(data, 1)
    assert isinstance(best_chromosome, BaseChromosome)
    assert isinstance(ea.data[0][0], dict)
    assert isinstance(ea.data[0][1], dict)
    assert isinstance(ea.data, list)


@pytest.mark.parametrize(
    "strategy, population, data",
    [
        ("mlp_strategy", "mlp_population", "mock_data"),
        ("conv2d_strategy", "conv_population", "mock_data"),
    ],
)
def test__population_asses(strategy, population, data, request):
    strategy = request.getfixturevalue(strategy)
    population = request.getfixturevalue(population)
    data = request.getfixturevalue(data)
    operations = request.getfixturevalue("operations")

    model_mock = mock.MagicMock()
    model_mock.evaluate = mock.MagicMock(return_value=(0.1, 0.2))

    ea = EvolutionaryAlgorithmElitism(
        strategy=strategy,
        pop_size=3,
        elitism_size=2,
        operations=operations,
    )
    ea.elite_models = [model_mock] * 2
    ea.get_model_fitness = mock.MagicMock(return_value=model_mock)

    ea.population = population
    models = ea._population_asses(data)
    assert isinstance(models, list)
    ea.get_model_fitness.assert_called_with(ea.population[0], data, strategy)


@pytest.mark.parametrize(
    "strategy, data",
    [
        ("mlp_strategy", "mock_data"),
        ("conv2d_strategy", "mock_data"),
    ],
)
def test_get_model_fitness(strategy, data, request):
    strategy = request.getfixturevalue(strategy)
    data = request.getfixturevalue(data)
    operations = request.getfixturevalue("operations")

    model_mock = mock.MagicMock()
    chromosome = mock.MagicMock()
    chromosome.get_fitness.return_value = (model_mock, 0.1, 0.2)

    ea = EvolutionaryAlgorithmElitism(
        strategy=strategy,
        pop_size=3,
        elitism_size=2,
        operations=operations,
    )

    model = ea.get_model_fitness(chromosome, data, strategy)
    assert model == model_mock
    assert chromosome.loss == 0.1
    assert chromosome.metric == 0.2


@pytest.mark.parametrize(
    "strategy, data",
    [
        ("mlp_strategy", "mock_data"),
        ("conv2d_strategy", "mock_data"),
    ],
)
def test_get_elite_model_fitness(strategy, data, request):
    strategy = request.getfixturevalue(strategy)
    data = request.getfixturevalue(data)
    operations = request.getfixturevalue("operations")

    model_mock = mock.MagicMock()
    model_mock.evaluate.return_value = (0.1, 0.2)
    chromosome = mock.MagicMock()
    chromosome.get_parameter.return_value = 64

    ea = EvolutionaryAlgorithmElitism(
        strategy=strategy,
        pop_size=3,
        elitism_size=2,
        operations=operations,
    )
    ea.elite_models = [model_mock]

    ea.get_elite_model_fitness(0, chromosome, data)
    assert chromosome.loss == 0.1
    assert chromosome.metric == 0.2


@pytest.mark.parametrize(
    "strategy, mock_data",
    [
        ("mlp_strategy", "mock_data"),
        ("conv2d_strategy", "mock_data"),
    ],
)
def test_fit(strategy, mock_data, request):
    strategy = request.getfixturevalue(strategy)
    data = request.getfixturevalue(mock_data)
    operations = request.getfixturevalue("operations")

    chromosome = mock.MagicMock()
    ea = EvolutionaryAlgorithmElitism(
        strategy=strategy,
        pop_size=3,
        elitism_size=2,
        operations=operations,
    )

    ea.strategy.generate_population = mock.MagicMock(
        return_value=[chromosome] * ea.pop_size
    )
    ea.evolve_population = mock.MagicMock(return_value=chromosome)
    ea.fit(mock_data, 1)

    ea.strategy.generate_population.assert_called_with(ea.pop_size)
    ea.evolve_population.assert_called_with(data, 0)
