"""
File containing base class for the parameter grids:
    one method and two properties: get method, and learnable_properties and static properties
"""

from dataclasses import dataclass


@dataclass
class ParameterGrid:
    """
    Base class that implements a grid  of learnable parameters of the model.

    """

    def get(self, parameter_name: str):
        """
        Return the value of the specified learnable parameter.

        :param parameter_name: str, the name of the parameter to get
        :return: value of the specified parameter
        """
        return getattr(self, parameter_name)

    @property
    def learnable_parameters(self):
        """
        Return a list of learnable parameters, where each element is a tuple
        containing the name of the parameter and its value(s).

        return: list of learnable parameters
        """
        return {
            key: item for key, item in self.__dict__.items() if isinstance(item, list)
        }

    @property
    def static_parameters(self):
        """
        Return a list of static parameters, where each element is a tuple
        containing the name of the parameter and its value.

        :return: list of static parameters
        """
        return {
            key: item
            for key, item in self.__dict__.items()
            if not isinstance(item, list)
        }
