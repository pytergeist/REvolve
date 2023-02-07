from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from abc import ABC, abstractmethod
from typing import Any, Callable
import random


class BaseStrategy(ABC):
    """
    This is an abstract base class for creating and generating a population of chromosomes.


    Abstract methods:
        create_new_chromosome(genes: list)
        generate_population(population_size: int)
    """

    @abstractmethod
    def create_new_chromosome(self, genes: list):  # pragma: no cover
        """
        Create a new chromosome from a list of genes.

        Args:
            genes (list): A list of genes to create a new chromosome from.

        Returns:
            None
        """
        pass

    def parameter_choice(self, parameter: Any):
        """
        Choose a random parameter from a list of available parameters.

        Args:
            parameter (Any): A parameter from which to choose from.

        Returns:
            Union[Any, int]: Returns a random parameter from a list, otherwise the original parameter.
        """
        params = self.parameters.get(parameter)

        if isinstance(params, list):
            return random.choice(params)
        else:
            return params

    @abstractmethod
    def generate_population(self, population_size: int):  # pragma: no cover
        """
        Generate a population of chromosomes.

        Args:
            population_size (int): The number of chromosomes to generate.

        Returns:
            None
        """
        pass

    @staticmethod
    def check_valid_architecture(chromosome, layer_param):
        """
        Check if the given chromosome architecture is valid, by ensuring the first layer
        of the chromosome does not have 0 value for the logits/filters, ensuring valid decoding
        of the chromosome

        Parameters:
        chromosome (Chromosome): The chromosome to check for validity.

        Returns:
        bool: True if the chromosome architecture is valid, False otherwise.
        """
        return getattr(chromosome.genes[0], layer_param) != 0

    def conv_block(self, gene: Callable, max_conv: int):
        """
        Create a convolutional block of layers.

        Args:
            gene (Callable): A callable function for creating a gene.
            max_conv (int): The number of convolutional layers to create.

        Returns:
            list: A list of convolutional layers.
        """
        return [
            gene(
                filters=self.parameter_choice("filters"),
                kernel_size=self.parameter_choice("kernel_size"),
                stride=self.parameter_choice("stride"),
                activation=self.parameter_choice("activation"),
            )
            for _ in range(max_conv)
        ]

    @staticmethod
    def squeeze_fc_neurons(fc_block):
        return sorted(fc_block, key=lambda x: x.hidden_neurons, reverse=True)

    @staticmethod
    def expand_conv_filters(conv_block):
        return sorted(conv_block, key=lambda x: x.filters, reverse=True)

    def fc_block(self, gene: Callable, max_fc: int):
        """
        Create a fully connected block of layers.

        Args:
            gene (Callable): A callable function for creating a gene.
            max_fc (int): The number of fully connected layers to create.

        Returns:
            list: A list of fully connected layers.
        """
        return [
            gene(
                hidden_neurons=self.parameter_choice("hidden_neurons"),
                activation=self.parameter_choice("activation"),
                dropout=self.parameter_choice("dropout"),
                l1=self.parameter_choice("l1"),
                l2=self.parameter_choice("l2"),
            )
            for _ in range(max_fc)
        ]

    def parameter_block(self, gene: Callable):
        """
        Create a block of training parameters.

        Args:
            gene (Callable): A callable function for creating a gene.

        Returns:
            list: A list of training parameters.
        """
        training_params = [
            "batch_size",
            "optimizer",
            "learning_rate",
        ]

        return [
            gene(parameter_name=param, parameter=self.parameter_choice(param))
            for param in training_params
        ]
