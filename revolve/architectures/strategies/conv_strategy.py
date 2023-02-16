"""
File containing Conv2DChromosome class:
    Conv2DChromosome represents the architecture of an 2d convolution network, including fully
    connected layer, 2d convolution layer and parameter genes
"""
from __future__ import annotations
import itertools
from typing import (
    List,
    Union,
    Callable,
)

import tensorflow as tf

from revolve.grids import ConvParameterGrid
from revolve.architectures.genes import Conv2DGene, FCGene, ParameterGene
from revolve.architectures.chromosomes import Conv2DChromosome
from revolve.architectures.base import BaseStrategy


class Conv2DStrategy(BaseStrategy):
    """
    Strategy class for handling Conv2D chromosomes. This strategy is responsible for
    generating a population of chromosomes, and checking if a chromosome is a
    valid architecture.

    Args:
        parameters (dataclass): Dataclass containing learnable parameters.
        max_fc (int, optional): Maximum number of fully connected layers. Defaults to 3.
        max_conv (int, optional): Maximum number of Conv2D layers. Defaults to 3.
        epochs (int, optional): Number of epochs for training. Defaults to 100.
        callback (tf.keras.callbacks.Callback, optional): Keras Callback object. Defaults to None.
        loss (Union[tf.keras.losses.Loss, str], optional): Loss function.
            Defaults to tf.keras.losses.MeanSquaredError().
        metric (Union[tf.keras.metrics.Metric, str], optional): Metric for evaluation.
            Defaults to tf.keras.metrics.MeanAbsoluteError().

    """

    # pylint: disable=too-many-instance-attributes

    def __init__(  # pylint: disable=too-many-arguments
        self,
        parameters: type[ConvParameterGrid],
        max_fc: int = 3,
        squeeze_fc: bool = False,
        max_conv: int = 3,
        expand_conv: bool = False,
        epochs: int = 100,
        callback: tf.keras.callbacks.Callback = None,
        loss: Union[tf.keras.losses.Loss, str] = tf.keras.losses.MeanSquaredError(),
        metric: Union[
            tf.keras.metrics.Metric, str
        ] = tf.keras.metrics.MeanAbsoluteError(),
    ):
        self.max_fc = max_fc
        self.squeeze_fc = squeeze_fc
        self.max_conv = max_conv
        self.expand_conv = expand_conv
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
        """
        Create a new Conv2DChromosome from a list of genes.

        Args:
            genes (List[Union[Conv2DGene, FCGene, ParameterGene]]): List of genes to be
            used for creating the chromosome.

        Returns:
            Conv2DChromosome: A new Conv2DChromosome.
        """
        return Conv2DChromosome(genes=genes)

    def conv_block(
        self, parameters: type[ConvParameterGrid], gene: Callable, max_conv: int
    ):
        """
        Create a convolutional block of layers.

        Args:
            parameters (dict): dictionary of learnable parameters
            gene (Callable): A callable function for creating a gene.
            max_conv (int): The number of convolutional layers to create.

        Returns:
            list: A list of convolutional layers.
        """
        return [
            gene(
                filters=self.parameter_choice(parameters, "filters"),
                kernel_size=self.parameter_choice(parameters, "kernel_size"),
                stride=self.parameter_choice(parameters, "stride"),
                activation=self.parameter_choice(parameters, "activation"),
            )
            for _ in range(max_conv)
        ]

    def generate_population(self, population_size: int) -> List:
        """
        Generate a population of chromosomes with unique architectures.

        Parameters:
        population_size (int): The size of the population to generate.

        Returns:
        List: The generated population of chromosomes.
        """
        assert isinstance(population_size, int)

        population: List[Conv2DChromosome] = []

        key_store: List[str] = []

        while len(population) < population_size:
            fc_block = self.fc_block(self.parameters, gene=FCGene, max_fc=self.max_fc)

            if self.squeeze_fc:
                fc_block = self.squeeze_fc_neurons(fc_block)

            conv_block = self.conv_block(
                self.parameters, gene=Conv2DGene, max_conv=self.max_conv
            )

            if self.expand_conv:
                conv_block = self.expand_conv_filters(conv_block)

            parameter_block = self.parameter_block(self.parameters, gene=ParameterGene)

            genes = list(itertools.chain(conv_block, fc_block, parameter_block))

            chromosome = self.create_new_chromosome(genes)

            key = chromosome.get_unique_key(chromosome.genes)

            if key not in key_store:
                if self.check_valid_architecture(
                    chromosome, "filters"
                ) and self.check_first_layer(chromosome, "filters"):
                    population.append(chromosome)
                    key_store.append(key)

        return population
