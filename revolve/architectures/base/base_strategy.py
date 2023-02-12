"""
File containing BaseChromosome class with abstract and defined method:
    each chromosome inherits BaseChromosome and every child chromosome
    represents the architecture of a network, including layers and parameter
    genes
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Union, TYPE_CHECKING
import random

if TYPE_CHECKING:
    from revolve.architectures.genes import FCGene, Conv2DGene
    from revolve.architectures.chromosomes import MLPChromosome, Conv2DChromosome
    from revolve.grids import MLPParameterGrid, ConvParameterGrid


class BaseStrategy(ABC):
    """
    This is an abstract base class for creating and generating a population of chromosomes.


    Abstract methods:
        create_new_chromosome(genes: list)
        generate_population(population_size: int)
    """

    @staticmethod
    @abstractmethod
    def create_new_chromosome(genes: list):  # pragma: no cover
        """
        Create a new chromosome from a list of genes.

        Args:
            genes (list): A list of genes to create a new chromosome from.

        Returns:
            None
        """

    @staticmethod
    def parameter_choice(
        parameters: type[Union[MLPParameterGrid, ConvParameterGrid]], parameter: Any
    ):
        """
        Choose a random parameter from a list of available parameters.

        Args:
            parameters (dict): A dictionary of learnable and static parameters'
            parameter (Any): A parameter from which to choose from.

        Returns:
            Union[Any, int]: Returns a random parameter from a list, otherwise the
            original parameter.
        """
        params = parameters.get(parameter)

        if isinstance(params, list):
            return random.choice(params)

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

    @staticmethod
    def check_valid_architecture(
        chromosome: Union[MLPChromosome, Conv2DChromosome], layer_param: str
    ):
        """
        Check if the given chromosome architecture is valid, by ensuring the first layer
        of the chromosome does not have 0 value for the logits/filters, ensuring valid decoding
        of the chromosome

        Args:
            chromosome (Chromosome): The chromosome to check for validity.
            layer_param (str): name of layer parameter to check  ('hidden_neurons', 'filters')

        Returns:
            bool: True if the chromosome architecture is valid, False otherwise.
        """
        test_layers = []
        for gene in chromosome.genes:
            if hasattr(gene, layer_param):
                test_layers.append(getattr(gene, layer_param) != 0)

        return any(test_layers)

    @staticmethod
    def check_first_layer(
        chromosome: Union[MLPChromosome, Conv2DChromosome], layer_param: str
    ):
        """
        Check first layer layer_param (logits/filters) is not 0:
            return True is this condition holds
        """
        return getattr(chromosome.genes[0], layer_param) != 0

    @staticmethod
    def squeeze_fc_neurons(fc_block: List[FCGene]):
        """
        Constrain fully connected layer block such that h_0>h_2>...>h_n
        where h_n = number of logits on hidden layer n

        Args:
            fc_block: List[FCGene]

        Returns:
            fc_block: List[FCGene] sorted in descending order
        """

        return sorted(fc_block, key=lambda x: x.hidden_neurons, reverse=True)

    @staticmethod
    def expand_conv_filters(conv_block: List[Conv2DGene]):
        """
        Constrain convolution layer block such that c_0<c_2>...>c_n
        where c_n = number of filters on hidden layer n

        Args:
            conv_block: List[Conv2DGene]

        Returns:
            conv_block: List[Conv2DGene] sorted in ascending order
        """
        return sorted(conv_block, key=lambda x: x.filters, reverse=True)

    def fc_block(
        self,
        parameters: type[Union[MLPParameterGrid, ConvParameterGrid]],
        gene: Callable,
        max_fc: int,
    ):
        """
        Create a fully connected block of layers.

        Args:
            parameters (dict): dictionary of learnable parameters'
            gene (Callable): A callable function for creating a gene.
            max_fc (int): The number of fully connected layers to create.

        Returns:
            list: A list of fully connected layers.
        """
        return [
            gene(
                hidden_neurons=self.parameter_choice(parameters, "hidden_neurons"),
                activation=self.parameter_choice(parameters, "activation"),
                dropout=self.parameter_choice(parameters, "dropout"),
                l1=self.parameter_choice(parameters, "l1"),
                l2=self.parameter_choice(parameters, "l2"),
            )
            for _ in range(max_fc)
        ]

    def parameter_block(
        self,
        parameters: type[Union[MLPParameterGrid, ConvParameterGrid]],
        gene: Callable,
    ):
        """
        Create a block of training parameters.

        Args:
            parameters (dict): dictionary of learnable parameters
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
            gene(
                parameter_name=param, parameter=self.parameter_choice(parameters, param)
            )
            for param in training_params
        ]
