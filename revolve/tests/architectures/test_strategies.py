import pytest
from unittest import mock
from revolve.architectures import MLPStrategy, MLPChromosome, Conv2DChromosome
from revolve.architectures import FCGene, ParameterGene


@pytest.mark.parametrize(
    "params, strategy_params",
    [
        (
            "mlp_params",
            {
                "max_fc": 1,
                "callback": mock.MagicMock(),
                "loss": "mean_squared_error",
                "metric": "mean_absolute_error",
            },
        ),
        (
            "mlp_params",
            {
                "max_fc": 1,
                "callback": mock.MagicMock(),
                "loss": mock.MagicMock(),
                "metric": mock.MagicMock(),
            },
        ),
    ],
)
def test_strategy_init(params, strategy_params, request):
    params = request.getfixturevalue(params)
    mlp_strategy = MLPStrategy(
        parameters=params,
        max_fc=strategy_params["max_fc"],
        callback=strategy_params["callback"],
        loss=strategy_params["loss"],
        metric=strategy_params["metric"],
    )
    assert mlp_strategy.max_fc == strategy_params["max_fc"]
    assert mlp_strategy.callback == strategy_params["callback"]
    assert mlp_strategy.loss == strategy_params["loss"] or mock.MagicMock()
    assert mlp_strategy.metric == strategy_params["metric"] or mock.MagicMock()
    assert mlp_strategy.parameters == params


@pytest.mark.parametrize(
    "strategy, chromosome_genes, expected_chromosome",
    [
        ("mlp_strategy", "mlp_chromosome_genes", MLPChromosome),
        ("conv2d_strategy", "conv_chromosome_genes", Conv2DChromosome)
    ],
)
def test_create_new_chromosome(
    strategy, chromosome_genes, expected_chromosome, request
):
    strategy = request.getfixturevalue(strategy)
    chromosome_genes = request.getfixturevalue(chromosome_genes)
    new_chromosome = strategy.create_new_chromosome(genes=chromosome_genes)
    assert isinstance(new_chromosome, expected_chromosome)


@pytest.mark.parametrize(
    "strategy, param",
    [
        ("mlp_strategy", "batch_size"),
        ("mlp_strategy", "regression_activation"),
    ],
)
def test_parameter_choice(strategy, param, request):
    strategy = request.getfixturevalue(strategy)
    chosen_param = strategy.parameter_choice(param)
    assert chosen_param in strategy.parameters[param]


@pytest.mark.parametrize(
    "strategy, population_size, expected_population_size",
    [
        ("mlp_strategy", 100, 100),  # Test with valid input
        ("mlp_strategy", 0, 0),  # Test with input 0
    ],
)
def test_generate_population(
    strategy, population_size, expected_population_size, request
):
    strategy = request.getfixturevalue(strategy)
    population = strategy.generate_population(population_size)
    assert len(population) == expected_population_size
    for chromosome in population:
        assert isinstance(chromosome, MLPChromosome)
        for gene in chromosome.genes:
            assert isinstance(gene, (FCGene, ParameterGene))
