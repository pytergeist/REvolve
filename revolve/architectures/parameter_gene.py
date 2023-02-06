from typing import Any
from revolve.architectures.base import Gene


class ParameterGene(Gene):
    def __init__(self, parameter_name: str, parameter: Any):
        super().__init__(
            gene_type=parameter_name,
            parameter=parameter,
        )
        self.parameters[parameter_name] = self.parameters.pop("parameter")
        self.__dict__.update(**self.parameters)

        self._validate_params()

    def _validate_params(self):
        pass
