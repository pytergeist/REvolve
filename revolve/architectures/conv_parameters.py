from dataclasses import dataclass
from typing import Union, List


@dataclass
class ConvParameterGrid:
    # possible learnable architecture parameter definitions
    filters: Union[List[int], int]
    kernel_size: Union[List[int], int]
    stride: Union[List[int], int]

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
