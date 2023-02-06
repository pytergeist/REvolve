import itertools
from typing import (
    List,
    Union,
)
from .conv2d_gene import Conv2DGene
from . import ParameterGene
from .fc_gene import FCGene
from .conv2d_chromosome import Conv2DChromosome
from revolve.architectures.base import Strategy


class Conv2DStrategy(Strategy):
    def __init__(
        self,
        input_shape: tuple,
        hidden_neurons: List[int],
        activation: Union[List[str], str],
        l1: Union[List[float], float],
        l2: Union[List[float], float],
        max_fc: int,
        filters: Union[List[float], float],
        kernel_size: Union[List[float], float],
        stride: Union[List[float], float],
        max_conv: int,
        dropout: Union[List[float], float],
        batch_size: Union[List[int], int],
        optimizer: Union[List[str], str],
        learning_rate: Union[List[float], float],
        callback: object,
        regression_target: int = 1,
        regression_activation: int = "relu",
    ):
        self.input_shape = input_shape
        self.neurons = hidden_neurons
        self.activation = activation
        self.l1 = l1
        self.l2 = l2
        self.max_fc = max_fc
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_conv = max_conv
        self.dropout = dropout
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.regression_target = regression_target
        self.regression_activation = regression_activation
        self.callback = callback
        self.learnable_parameters = self.get_learnable_parameters()

    @staticmethod
    def create_new_chromosome(genes: List[Union[Conv2DGene, FCGene, ParameterGene]]):
        return Conv2DChromosome(genes=genes)

    def generate_population(self, population_size: int) -> List:
        assert isinstance(population_size, int)

        population: List[Conv2DChromosome] = []

        key_store: List[str] = []

        while len(population) < population_size:
            fc_block = self.fc_block(gene=FCGene, max_fc=self.max_fc)

            conv_block = self.conv_block(
                gene=Conv2DGene, gene_type="conv2d", max_conv=self.max_conv
            )

            parameter_block = self.parameter_block(gene=ParameterGene)

            genes = list(itertools.chain(conv_block, fc_block, parameter_block))

            chromosome = Conv2DChromosome(genes)

            key = chromosome.get_unique_key(chromosome.genes)

            if key not in key_store:
                population.append(chromosome)
                key_store.append(key)

        return population
