from dataclasses import dataclass
from typing import Union, List


@dataclass
class FCParameterGrid:
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

    # always static parameter definitions
    input_shape: tuple
    regression_target: int = 1
    regression_activation: str = "relu"

    def get(self, parameter_name):
        return getattr(self, parameter_name)

    @property
    def learnable_parameters(self):
        return [
            (key, self.__dict__[key])
            for key in self.__dict__
            if isinstance(self.__dict__[key], list)
        ]

    @property
    def static_parameters(self):
        return [
            (key, self.__dict__[key])
            for key in self.__dict__
            if not isinstance(self.__dict__[key], list)
        ]
