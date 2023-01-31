import numpy as np
import pytest
from revolve.architectures import (
    FCGene, ParameterGene,
    MLPStrategy, MLPChromosome,
    Conv2DChromosome, Conv2DGene,
    Conv2DStrategy
)
from revolve.algorithms import EvolutionaryAlgorithmElitism


@pytest.fixture
def data():
    x_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    x_test = np.random.rand(50, 10)
    y_test = np.random.rand(50)
    return x_train, y_train, x_test, y_test


@pytest.fixture
def mlp_learnable_parameters():
    return {
        'neurons': [64, 128, 256],
        'activation': ['relu', 'swish'],
        'dropout': [0.1, 0.3, 0.5],
        'batch_size': [128, 256, 512],
        'optimizer': ['adam', 'sgd'],
        'learning_rate': [1e-3, 5e-3, 7.5e-3],
        'l1': [0.001, 0.01, 0.1],
        'l2': [0.001, 0.01, 0.1],
    }


@pytest.fixture
def conv2d_learnable_parameters(mlp_learnable_parameters):
    conv2d_param = {
        'filters': [16, 32, 64],
        'kernel_size': [1, 2],
        'stride': [1, 2],
        'activation': ['relu', 'swish'],
    }

    return {**mlp_learnable_parameters, **conv2d_param}



@pytest.fixture(scope='class')
def fc_gene():
    return FCGene(gene_type='fc', neurons=128, activation='relu', dropout=0.5, l1=0.01, l2=0.01)


@pytest.fixture(scope='class')
def conv2d_gene():
    return Conv2DGene(gene_type='conv2d', filters=16, kernel_size=2, stride=1, activation='relu')


@pytest.fixture
def gene_list_mlp():
    return [
        FCGene(gene_type='fc', neurons=128, activation='relu', dropout=0.2, l1=0.01, l2=0.01),
        FCGene(gene_type='fc', neurons=64, activation='relu', dropout=0.2, l1=0.01, l2=0.01),
        FCGene(gene_type='fc', neurons=32, activation='relu', dropout=0.2, l1=0.01, l2=0.01),
        ParameterGene(parameter_name='optimizer', parameter='adam'),
        ParameterGene(parameter_name='learning_rate', parameter=1e-3),
    ]


@pytest.fixture
def gene_list_conv2d():
    return [
        Conv2DGene(gene_type='conv2d', filters=16, kernel_size=2, stride=1, activation='relu'),
        Conv2DGene(gene_type='conv2d', filters=32, kernel_size=1, stride=1, activation='relu'),
        Conv2DGene(gene_type='conv2d', filters=32, kernel_size=1, stride=1, activation='relu'),
        FCGene(gene_type='fc', neurons=128, activation='relu', dropout=0.2, l1=0.01, l2=0.01),
        FCGene(gene_type='fc', neurons=64, activation='relu', dropout=0.2, l1=0.01, l2=0.01),
        FCGene(gene_type='fc', neurons=32, activation='relu', dropout=0.2, l1=0.01, l2=0.01),
        ParameterGene(parameter_name='optimizer', parameter='adam'),
        ParameterGene(parameter_name='learning_rate', parameter=1e-3),
    ]


@pytest.fixture(scope='class')
def mlp_strategy():
    input_shape = 10
    hidden_neurons = [10, 20, 30]
    activation = 'relu'
    l1 = [0.01, 0.02]
    l2 = [0.01, 0.02]
    max_fc = 2
    dropout = [0.2, 0.3]
    batch_size = [32, 64]
    optimizer = ['adam', 'sgd']
    learning_rate = [0.01, 0.001]
    callback = [],
    regression_target = 1
    regression_activation = 'relu'
    return MLPStrategy(
        input_shape=input_shape, hidden_neurons=hidden_neurons,
        activation=activation, l1=l1, l2=l2, max_fc=max_fc,
        dropout=dropout, batch_size=batch_size, optimizer=optimizer,
        learning_rate=learning_rate, callback=callback,
        regression_target=regression_target, regression_activation=regression_activation,
    )


@pytest.fixture(scope='class')
def conv2d_strategy():
    input_shape = (10, 10, 1)
    hidden_neurons = [10, 20, 30]
    activation = 'relu'
    l1 = [0.01, 0.02]
    l2 = [0.01, 0.02]
    max_fc = 2
    filters = [32, 64]
    kernel_size = [1, 2]
    stride = [1, 2]
    max_conv = 2
    dropout = [0.2, 0.3]
    batch_size = [32, 64]
    optimizer = ['adam', 'sgd']
    learning_rate = [0.01, 0.001]
    callback = [],
    regression_target = 1
    regression_activation = 'relu'
    return Conv2DStrategy(
        input_shape=input_shape, hidden_neurons=hidden_neurons,
        activation=activation, l1=l1, l2=l2, max_fc=max_fc,
        filters=filters, kernel_size=kernel_size, stride=stride, max_conv=max_conv,
        dropout=dropout, batch_size=batch_size, optimizer=optimizer,
        learning_rate=learning_rate, callback=callback,
        regression_target=regression_target, regression_activation=regression_activation,
    )

@pytest.fixture
def generation_data():
    return [
        (1, 0.1),
        (2, 0.2),
        (3, 0.3),
        (4, 0.4),
        (5, 0.5),
        (6, 0.6),
        (7, 0.7),
        (8, 0.8),
        (9, 0.9),
        (10, 1.0)
    ]


@pytest.fixture
def mlp_models():
    return [
        MLPChromosome(genes=[1, 2, 3]),
        MLPChromosome(genes=[4, 5, 6]),
        MLPChromosome(genes=[7, 8, 9]),
        MLPChromosome(genes=[10, 11, 12]),
        MLPChromosome(genes=[13, 14, 15]),
        MLPChromosome(genes=[16, 17, 18]),
        MLPChromosome(genes=[19, 20, 21]),
        MLPChromosome(genes=[22, 23, 24]),
        MLPChromosome(genes=[25, 26, 27]),
        MLPChromosome(genes=[28, 29, 30])
    ]


@pytest.fixture
def conv2d_models():
    return [
        Conv2DChromosome(genes=[1, 2, 3]),
        Conv2DChromosome(genes=[4, 5, 6]),
        Conv2DChromosome(genes=[7, 8, 9]),
        Conv2DChromosome(genes=[10, 11, 12]),
        Conv2DChromosome(genes=[13, 14, 15]),
        Conv2DChromosome(genes=[16, 17, 18]),
        Conv2DChromosome(genes=[19, 20, 21]),
        Conv2DChromosome(genes=[22, 23, 24]),
        Conv2DChromosome(genes=[25, 26, 27]),
        Conv2DChromosome(genes=[28, 29, 30])
    ]


class DummyModel:
    def compile(self):
        pass

    def fit(self):
        pass

    def evaluate(self):
        return 0.1, 0.9


class DummyChromosome:
    def __init__(self, genes=None):
        if genes:
            self.genes = genes
        else:
            self.genes = [
                [1, 2], [3, 4],
                [5, 6], [7, 8],
            ]


class DummyStrategy:
    def __init__(self):
        self.learnable_parameters = {'hidden_neurons': [1, 2, 3]}

    def generate_population(self, pop_size: int):
        return [DummyChromosome() for _ in range(pop_size)]

    def asses(self, x_train, y_train, x_test, y_test, chromosome, epochs):
        return DummyModel(), 0.1, 0.9

    def create_new_chromosome(self, genes):
        return DummyChromosome(genes=genes)


@pytest.fixture(scope='class')
def ga_elitism_mlp():
    dummy_strategy = DummyStrategy()
    ga = EvolutionaryAlgorithmElitism(
        strategy=dummy_strategy, pop_size=10,
        tournament_size=2, elitism_size=2,
        crossover_prob=0.5, mutation_prob=0.5
    )
    ga.population = ga.strategy.generate_population(20)
    return ga
