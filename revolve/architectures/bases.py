import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np
import numpy.typing as npt


class BaseGene(ABC):
    """
    Abstract class for chromosomes
        implements abstract methods for:
            - mutate
            - get_attributes
            - get_attribute_names
    """

    @abstractmethod
    def mutate(self, learnable_parameters: dict):
        pass

    @abstractmethod
    def get_attributes(self):
        pass

    @abstractmethod
    def get_attribute_names(self):
        pass


class BaseChromosome(ABC):
    """
    Abstract class for chromosomes
        implements abstract methods for:
            -  get_unique_key
            - get_parameter
            - decode
    """

    @abstractmethod
    def get_unique_key(self):
        pass

    @abstractmethod
    def get_parameter(self, param: str, default_param: Any):
        pass

    @abstractmethod
    def decode(self, input_shape: int, regression_target: int,
               regression_activation: str) -> tf.keras.Model:
        pass


class BaseStrategy(ABC):
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
    def create_new_chromosome(self, genes: list):
        pass

    @abstractmethod
    def get_learnable_parameters(self):
        pass

    @abstractmethod
    def parameter_choice(self, parameter: Any):
        pass

    @abstractmethod
    def generate_population(self, population_size: int):
        pass

    @abstractmethod
    def asses(self,
              x_train: npt.NDArray[np.float32],
              y_train: npt.NDArray[np.float32],
              x_test: npt.NDArray[np.float32],
              y_test: npt.NDArray[np.float32],
              chromosome: object,
              epochs: int
              ):
        pass
