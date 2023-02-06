from dataclasses import dataclass


@dataclass
class ParameterGrid:
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
