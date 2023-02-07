import itertools
from dataclasses import dataclass
from typing import Union, List

import tensorflow as tf

from revolve.architectures.genes import FCGene, ParameterGene
from revolve.architectures.chromosomes import MLPChromosome
from revolve.architectures.base import BaseStrategy


class MLPStrategy(BaseStrategy):
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

    @staticmethod  # no longer needed
    def create_new_chromosome(genes: List[object]):
        return MLPChromosome(genes=genes)

    @staticmethod
    def check_valid_architecture(chromosome):
        return chromosome.genes[0].hidden_neurons != 0

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

            if key not in key_store and self.check_valid_architecture(chromosome):
                population.append(chromosome)
                key_store.append(key)

        return population
