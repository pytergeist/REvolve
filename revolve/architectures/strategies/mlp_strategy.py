"""
File containing MLPChromosome class:
    MLPChromosome represents the architecture of an MLP network, including fully connected layer
    and parameter genes
"""
from __future__ import annotations
import itertools
from typing import Union, List

import tensorflow as tf

from revolve.grids import MLPParameterGrid
from revolve.architectures.genes import FCGene, ParameterGene
from revolve.architectures.chromosomes import MLPChromosome
from revolve.architectures.base import BaseStrategy


class MLPStrategy(BaseStrategy):
    """
    Strategy class for handling MLP chromosomes. This strategy is responsible for
    generating a population of MLPChromosomes, and checking if a chromosome is a valid architecture.

    Args:
        parameters (dataclass): Dataclass containing learnable parameters.
        max_fc (int, optional): Maximum number of fully connected layers. Defaults to 3.
        epochs (int, optional): Number of epochs for training. Defaults to 100.
        callback (tf.keras.callbacks.Callback, optional): Keras Callback object. Defaults to None.
        loss (Union[tf.keras.losses.Loss, str], optional): Loss function.
            Defaults to tf.keras.losses.MeanSquaredError().
        metric (Union[tf.keras.metrics.Metric, str], optional): Metric for evaluation.
            Defaults to tf.keras.metrics.MeanAbsoluteError().

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        parameters: type[MLPParameterGrid],
        max_fc: int = 3,
        squeeze_fc: bool = False,
        epochs: int = 100,
        callback: tf.keras.callbacks.Callback = None,
        loss: Union[tf.keras.losses.Loss, str] = tf.keras.losses.MeanSquaredError(),
        metric: Union[
            tf.keras.metrics.Metric, str
        ] = tf.keras.metrics.MeanAbsoluteError(),
    ):
        self.max_fc = max_fc
        self.squeeze_fc = squeeze_fc
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
    def create_new_chromosome(genes: List[Union[FCGene, ParameterGene]]):
        """
        Create a new MLPChromosome from a list of genes.

        Args:
            genes (List[Union[FCGene, ParameterGene]]): List of genes to be used for
            creating the chromosome.

        Returns:
            MLPChromosome: A new MLPChromosome.
        """
        return MLPChromosome(genes=genes)

    def generate_population(self, population_size: int) -> List:
        """
        Generate a population of chromosomes with unique architectures.

        Parameters:
        population_size (int): The size of the population to generate.

        Returns:
        List: The generated population of chromosomes.
        """
        assert isinstance(population_size, int)

        population: List[MLPChromosome] = []

        key_store: List[str] = []

        while len(population) < population_size:
            fc_block = self.fc_block(self.parameters, FCGene, max_fc=self.max_fc)

            if self.squeeze_fc:
                fc_block = self.squeeze_fc_neurons(fc_block)

            parameter_block = self.parameter_block(self.parameters, gene=ParameterGene)

            genes = list(itertools.chain(fc_block, parameter_block))

            chromosome = self.create_new_chromosome(genes)

            key = chromosome.get_unique_key(chromosome.genes)

            if key not in key_store:
                if self.check_valid_architecture(
                    chromosome, "hidden_neurons"
                ) and self.check_first_layer(chromosome, "hidden_neurons"):
                    population.append(chromosome)
                    key_store.append(key)

        return population
