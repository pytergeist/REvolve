import itertools
from dataclasses import dataclass
from typing import Union, List

import tensorflow as tf
import tensorflow_addons as tfa

from revolve.architectures import FCGene, ParameterGene
from .mlp_chromosome import MLPChromosome
from revolve.architectures.base import Strategy


class MLPStrategy(Strategy):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        parameters: dataclass,
        max_fc: int = 3,
        epochs: int = 100,
        callback: tf.keras.callbacks.Callback = None,
        loss: Union[tf.keras.losses.Loss, str] = tf.keras.losses.MeanSquaredError(),
        metric: Union[
            tf.keras.metrics.Metric, str
        ] = tf.keras.metrics.MeanAbsoluteError(),
    ):
        self.max_fc = max_fc
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
    def create_new_chromosome(genes: List[object]):
        return MLPChromosome(genes=genes)

    def generate_population(self, population_size: int) -> List:
        assert isinstance(population_size, int)

        population: List[MLPChromosome] = []

        key_store: List[str] = []

        while len(population) < population_size:
            fc_block = self.fc_block(gene=FCGene, max_fc=self.max_fc)

            parameter_block = self.parameter_block(gene=ParameterGene)

            genes = list(itertools.chain(fc_block, parameter_block))

            chromosome = MLPChromosome(genes)

            key = chromosome.get_unique_key(chromosome.genes)

            if key not in key_store:
                population.append(chromosome)
                key_store.append(key)

        return population
