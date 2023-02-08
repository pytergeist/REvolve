"""
File containing class for MLPParameterGrid, child class of ParameterGrid:
    defines values for learnable and static parameters of MLP
"""

from dataclasses import dataclass
from typing import Union, List
from .base import ParameterGrid


@dataclass
class MLPParameterGrid(ParameterGrid):  # pylint: disable=too-many-instance-attributes
    """
    Subclass of Parameter grid that holds learnable/static hyper-parameters
    for multi-layer perceptrons
    """

    # always static parameter definitions
    input_shape: tuple
    regression_target: int
    regression_activation: str

    # possible learnable architecture parameter definitions
    hidden_neurons: Union[List[int], int]
    activation: Union[List[str], str]
    l1: Union[List[float], float]
    l2: Union[List[float], float]
    dropout: Union[List[float], float]

    # possible learnable training parameter definitions
    batch_size: Union[List[int], int]
    optimizer: Union[List[str], str]
    learning_rate: Union[List[float], float]
