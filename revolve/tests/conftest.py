import pytest
import itertools
from unittest import mock
from revolve.architectures.chromosomes import MLPChromosome, Conv2DChromosome
from revolve.architectures.strategies import MLPStrategy, Conv2DStrategy
from revolve.architectures.genes import FCGene, Conv2DGene, ParameterGene
from revolve.architectures.base import BaseChromosome


@pytest.fixture
def parameter_gene_params():
    return {
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "batch_size": 32,
    }


@pytest.fixture
def parameter_learnable_params():
    return {
        "learning_rate": [1e-4, 1e-5],
        "optimizer": ["sgd", "rmsprop"],
        "batch_size": [16, 64],
    }


@pytest.fixture
def fc_gene_params():
    return {
        "hidden_neurons": 100,
        "activation": "relu",
        "dropout": 0.5,
        "l1": 0.1,
        "l2": 0.2,
    }


@pytest.fixture
def fc_learnable_params():
    return {
        "hidden_neurons": [200, 300],
        "activation": ["linear", "swish"],
        "dropout": [0.3, 0.4],
        "l1": [0.05, 0.06],
        "l2": [0.15, 0.16],
    }


@pytest.fixture
def conv_gene_params(fc_gene_params):
    return {
        "filters": 16,
        "kernel_size": 2,
        "stride": 1,
        "activation": "relu",
    }


@pytest.fixture
def mlp_params(fc_learnable_params, parameter_learnable_params):
    static_params = {
        "input_shape": (10,),
        "regression_target": 1,
        "regression_activation": "relu",
    }
    return {**static_params, **fc_learnable_params, **parameter_learnable_params}


@pytest.fixture
def conv_learnable_params():
    return {
        "filters": [32, 64],
        "kernel_size": [1, 3],
        "stride": [2, 3],
        "activation": ["swish", "tanh"],
    }


@pytest.fixture
def conv_network_params(mlp_params, conv_learnable_params):
    network_params = {**mlp_params, **conv_learnable_params, "input_shape": (10, 10, 1)}
    return network_params


@pytest.fixture
def mlp_strategy_params():
    return {
        "max_fc": 3,
        "squeeze_fc": True,
        "epochs": 10,
        "callback": mock.MagicMock(),
        "loss": mock.MagicMock(),
        "metric": mock.MagicMock(),
    }


@pytest.fixture
def conv_strategy_params():
    return {
        "max_conv": 3,
        "expand_conv": True,
        "max_fc": 2,
        "squeeze_fc": True,
        "epochs": 10,
        "callback": mock.MagicMock(),
        "loss": mock.MagicMock(),
        "metric": mock.MagicMock(),
    }


@pytest.fixture
def mlp_chromosome_genes(fc_gene_params, parameter_gene_params):
    fc_genes = [FCGene(**fc_gene_params) for _ in range(3)]
    param_genes = [
        ParameterGene(parameter_name=param_name, parameter=param)
        for param_name, param in parameter_gene_params.items()
    ]
    return list(itertools.chain(fc_genes, param_genes))


@pytest.fixture
def conv_chromosome_genes(fc_gene_params, conv_gene_params, parameter_gene_params):
    conv_genes = [Conv2DGene(**conv_gene_params) for _ in range(3)]
    fc_genes = [FCGene(**fc_gene_params) for _ in range(2)]
    param_genes = [
        ParameterGene(parameter_name=param_name, parameter=param)
        for param_name, param in parameter_gene_params.items()
    ]
    return list(itertools.chain(conv_genes, fc_genes, param_genes))


@pytest.fixture
def mlp_chromosome_fc_only_genes(fc_gene_params, parameter_gene_params):
    return [FCGene(**fc_gene_params) for _ in range(3)]


@pytest.fixture
def conv_chromosome_fc_conv_only_genes(
    fc_gene_params, conv_gene_params, parameter_gene_params
):
    conv_genes = [Conv2DGene(**conv_gene_params) for _ in range(3)]
    fc_genes = [FCGene(**fc_gene_params) for _ in range(3)]
    return list(itertools.chain(conv_genes, fc_genes))


@pytest.fixture
def mlp_chromosome(mlp_chromosome_genes):
    return MLPChromosome(genes=mlp_chromosome_genes)


@pytest.fixture
def conv2d_chromosome(conv_chromosome_genes):
    return Conv2DChromosome(genes=conv_chromosome_genes)


@pytest.fixture
def mock_data():
    train_data = mock.MagicMock()
    valid_data = mock.MagicMock()
    test_data = mock.MagicMock()
    train_data.batch.return_value = "train_batch"
    valid_data.batch.return_value = "valid_batch"
    test_data.batch.return_value = "test_batch"
    return train_data, valid_data, test_data


@pytest.fixture
def mlp_strategy(mlp_params, mlp_strategy_params):
    return MLPStrategy(mlp_params, **mlp_strategy_params)


@pytest.fixture
def conv2d_strategy(conv_network_params, conv_strategy_params):
    return Conv2DStrategy(conv_network_params, **conv_strategy_params)


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
    operations.selection = mock.MagicMock(return_value=(BaseChromosome, BaseChromosome))
    operations.crossover = mock.MagicMock(return_value=BaseChromosome)
    operations.mutation = mock.MagicMock(return_value=BaseChromosome)
    return operations


if __name__ == "__main__":
    pytest.main()
