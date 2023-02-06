from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from abc import ABC, abstractmethod
from typing import Any, Callable
import random


class Strategy(ABC):
    """
    Abstract class for strategies
        implements abstract methods for:
            - create_new_chromosome
            - get_learnable_parameters
            - parameter_choice
            - generate_population
            - asses
    """

    @abstractmethod
    def create_new_chromosome(self, genes: list):  # pragma: no cover
        pass

    def parameter_choice(self, parameter: Any):
        params = self.parameters.get(parameter)

        if isinstance(params, list):
            return random.choice(params)
        else:
            return params

    @abstractmethod
    def generate_population(self, population_size: int):  # pragma: no cover
        pass

    def conv_block(self, gene: Callable, gene_type: str, max_conv: int):
        return [
            gene(
                gene_type=gene_type,
                filters=self.parameter_choice("filters"),
                kernel_size=self.parameter_choice("kernel_size"),
                stride=self.parameter_choice("stride"),
                activation=self.parameter_choice("activation"),
            )
            for _ in range(max_conv)
        ]

    def fc_block(self, gene: Callable, max_fc: int):
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
        training_params = [
            "batch_size",
            "optimizer",
            "learning_rate",
        ]

        return [
            gene(parameter_name=param, parameter=self.parameter_choice(param))
            for param in training_params
        ]
