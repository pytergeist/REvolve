from typing import Any
from revolve.architectures.base import BaseGene


class ParameterGene(BaseGene):
    """
    ParameterGene is a sub-class of BaseGene which represents a parameter in an MLP/CONV model
    Each ParameterGene holds a single parameter

    :param parameter_name: (str) The name of the parameter
    :param parameter: (Any): the value of the parameter
    """

    def __init__(self, parameter_name: str, parameter: Any):
        super().__init__(
            gene_type=parameter_name,
            parameter=parameter,
        )
        self.parameters[parameter_name] = self.parameters.pop("parameter")
        self.__dict__.update(**self.parameters)

        self._validate_params()

    def _validate_params(self):
        """
        Internal method to validate the parameters passed to the constructor.
        """
        pass
