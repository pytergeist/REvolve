import itertools
import tensorflow as tf
from dataclasses import dataclass
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
    def __init__(  # pylint: disable=too-many-arguments
        self,
        parameters: dataclass,
        max_fc: int = 3,
        max_conv: int = 3,
        epochs: int = 100,
        callback: tf.keras.callbacks.Callback = None,
        loss: Union[tf.keras.losses.Loss, str] = tf.keras.losses.MeanSquaredError(),
        metric: Union[
            tf.keras.metrics.Metric, str
        ] = tf.keras.metrics.MeanAbsoluteError(),
    ):
        self.max_fc = max_fc
        self.max_conv = max_conv
        self.epochs = epochs
        self.callback = callback

        if isinstance(loss, str):
            self.loss = tf.keras.losses.get(loss)
        else:
            self.loss = loss

        if isinstance(metric, str):
            self.metric = tf.keras.metrics.get(metric)
        else:
            self.metric = metric

        self.parameters = parameters  # self.get_learnable_parameters()

    @staticmethod
    def create_new_chromosome(genes: List[Union[Conv2DGene, FCGene, ParameterGene]]):
        return Conv2DChromosome(genes=genes)

    def generate_population(self, population_size: int) -> List:
        assert isinstance(population_size, int)

        population: List[Conv2DChromosome] = []

        key_store: List[str] = []

        while len(population) < population_size:
            fc_block = self.fc_block(gene=FCGene, max_fc=self.max_fc)

            conv_block = self.conv_block(gene=Conv2DGene, max_conv=self.max_conv)

            parameter_block = self.parameter_block(gene=ParameterGene)

            genes = list(itertools.chain(conv_block, fc_block, parameter_block))

            chromosome = Conv2DChromosome(genes)

            key = chromosome.get_unique_key(chromosome.genes)

            if key not in key_store:
                population.append(chromosome)
                key_store.append(key)

        return population
