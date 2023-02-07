from dataclasses import dataclass
from typing import Union, List
from .base import ParameterGrid


@dataclass
class ConvParameterGrid(ParameterGrid):
    """
    Subclass of Parameter grid that holds learnable/static hyper-parameters
    for convolution neural networks
    """

    # always static parameter definitions
    input_shape: tuple
    regression_target: int
    regression_activation: str

    # possible learnable architecture parameter definitions
    filters: Union[List[int], int]
    kernel_size: Union[List[int], int]
    stride: Union[List[int], int]

    # possible learnable training parameter definitions
    hidden_neurons: Union[List[int], int]
    activation: Union[List[str], str]
    l1: Union[List[float], float]
    l2: Union[List[float], float]
    dropout: Union[List[float], float]

    # possible learnable training parameter definitions
    batch_size: Union[List[int], int]
    optimizer: Union[List[str], str]
    learning_rate: Union[List[float], float]
